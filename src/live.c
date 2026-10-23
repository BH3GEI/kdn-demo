/* 在 MacBook 上用真实模型跑分层 KV。
 *
 * 第一部分：同样长度的上下文，从头重算 / 从 DRAM 拷回 / 从 SSD 读回，各要多久。
 * 第二部分：从你的真实轨迹里截一段多用户的繁忙时段，按比例缩小 token 数，
 *          分别用 LRU 和基于信号的调度跑一遍，比较实测的首 token 延迟。
 *
 * 用法：./live [--model models/qwen3-1.7b-q4_k_m.gguf] [--users 4] [--calls 300]
 *              [--scale 40] [--gpu-frac 0.25] [--skip-bench]
 */
#include "kvengine.h"

static void rand_tokens(llama_token *out, int n) {
    for (int i = 0; i < n; i++) out[i] = 1000 + rng_next() % 149000;
}

static void microbench(struct llama_model *model) {
    static const int L[] = {2000, 8000, 16000, 32000};
    printf("\n== Part 1: time to get a context back (measured) ==\n");
    printf("%8s %11s %11s %11s %9s\n", "context", "recompute", "DRAM->GPU", "SSD->GPU", "KV size");
    kengine_t *e = ke_new(model, 36000, 1L << 40, 1L << 40, POL_LRU, 0, "data/kv_ssd_bench");
    llama_token *toks = malloc(32000 * sizeof(llama_token));
    for (int k = 0; k < 4; k++) {
        rand_tokens(toks, L[k]);
        ksess_t *s = ke_get(e, k, 0);
        e->plen[0] = 0;
        s->slot = ke_alloc_slot(e, 0);
        double t0 = ke_now();
        ke_prefill(e, s->slot, 0, toks, L[k]);
        double recompute = ke_now() - t0;
        s->hist = malloc(L[k] * sizeof(llama_token)); memcpy(s->hist, toks, L[k] * sizeof(llama_token));
        s->nhist = s->caphist = L[k];
        s->tier = T_GPU; e->used[T_GPU] += L[k];
        ke_demote(e, s, T_DRAM, 0);
        double mb = s->bufsz / 1e6;
        e->used[T_DRAM] -= L[k];
        double dram = ke_restore(e, s);
        e->used[T_GPU] += L[k];
        ke_demote(e, s, T_SSD, 0);
        e->used[T_SSD] -= L[k];
        double ssd = ke_restore(e, s);
        printf("%8d %10.3fs %10.3fs %10.3fs %7.0fMB\n", L[k], recompute, dram, ssd, mb);
        llama_memory_seq_rm(e->mem, s->slot, -1, -1);
        e->slot_free[e->n_free++] = s->slot; s->slot = -1; s->tier = T_NONE;
        e->used[T_GPU] = 0;
    }
    printf("(SSD reads use F_NOCACHE, bypassing the page cache; Qwen3-1.7B f16 KV is ~112KB/token)\n");
    free(toks);
    ke_free(e);
}

/* 找调用最密集的 2 小时，取其中前 max_calls 次调用 */
static int pick_window(const workload_t *w, int max_calls) {
    int best = 0, bj = 0, j = 0;
    for (int i = 0; i < w->n_ev; i++) {
        while (w->ev[i].t - w->ev[j].t > 7200) j++;
        if (i - j > best) { best = i - j; bj = j; }
    }
    (void)max_calls;
    return bj;
}

typedef struct { double gpu, dram, ssd, miss, mean, p50, p95, io, restore, prefill, pf_tok; long recompute; int n; } summary_t;

static int cmp_d(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return d < 0 ? -1 : d > 0;
}

static summary_t replay(struct llama_model *model, const workload_t *w, int ev0, int n_ev, int policy,
                        int dedup, int scale, long gpu_tok, long dram_tok, int n_ctx) {
    char dir[64];
    snprintf(dir, sizeof dir, "data/kv_ssd_%s", POL_NAME[policy]);
    kengine_t *e = ke_new(model, n_ctx, gpu_tok, dram_tok, policy, dedup, dir);
    rng_state = 12345;
    llama_token *buf = malloc(2 * 1000000 / scale * sizeof(llama_token) + 4096 * sizeof(llama_token));
    for (int p = 0; p < N_PREFIX; p++) {
        int n = SHARED_PREFIXES[p] / scale;
        rand_tokens(buf, n);
        ke_register_prefix(e, p, buf, n);
    }
    int *sid_of = malloc(w->n_inst * sizeof(int)), *have = calloc(w->n_inst + n_ev, sizeof(int));
    double *last_end = malloc(w->n_inst * sizeof(double));
    for (int i = 0; i < w->n_inst; i++) { sid_of[i] = -1; last_end[i] = -1; }
    int next_sid = 0;
    double *ttft = malloc(n_ev * sizeof(double));
    summary_t r = {0};
    for (int k = 0; k < n_ev; k++) {
        const event_t *ev = &w->ev[ev0 + k];
        const session_t *base = w->inst[ev->inst].base;
        double shift = w->inst[ev->inst].shift;
        const call_t *cl = &base->calls[ev->call];
        int prefix = prefix_index(base->shared);
        int target = (cl->ctx - base->shared) / scale;
        if (target < 1) target = 1;
        int sid = sid_of[ev->inst];
        if (sid < 0 || target < have[sid]) {   /* 新会话，或上下文被压缩（旧 KV 作废） */
            sid = sid_of[ev->inst] = next_sid++;
        }
        if (last_end[ev->inst] >= 0 && ev->call > 0)
            ke_observe(e, sid, base->calls[ev->call - 1].sig, cl->start - last_end[ev->inst]);
        int n = target - have[sid];
        if (n < 8) n = 8;
        rand_tokens(buf, n);
        kresult_t x = ke_request(e, sid, prefix, buf, n, cl->start + shift);
        have[sid] += n;
        double end = cl->end + shift;
        double next = ev->call + 1 < base->n ? base->calls[ev->call + 1].start + shift : end + 10 * DAY;
        ke_finish(e, sid, cl->sig, end, next);
        last_end[ev->inst] = cl->end;
        if (x.where == -2) continue;
        ttft[r.n++] = x.ttft_s;
        r.mean += x.ttft_s;
        r.recompute += x.recompute_tok;
        r.restore += x.restore_s; r.prefill += x.prefill_s; r.pf_tok += x.new_tok;
        if (x.where == T_GPU) r.gpu++;
        else if (x.where == T_DRAM) r.dram++;
        else if (x.where == T_SSD) r.ssd++;
        else r.miss++;
    }
    qsort(ttft, r.n, sizeof(double), cmp_d);
    r.pf_tok /= r.prefill; r.restore /= r.n; r.prefill /= r.n;
    r.mean /= r.n; r.p50 = ttft[r.n / 2]; r.p95 = ttft[(int)(0.95 * r.n)];
    r.gpu /= r.n; r.dram /= r.n; r.ssd /= r.n; r.miss /= r.n;
    r.io = e->io_s;
    free(buf); free(sid_of); free(have); free(last_end); free(ttft);
    ke_free(e);
    return r;
}

int main(int argc, char **argv) {
    const char *model_path = "models/qwen3-1.7b-q4_k_m.gguf", *trace = "data/trace.csv";
    int users = 4, calls = 300, scale = 40, bench = 1;
    double gpu_frac = 0.25;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--model") && a + 1 < argc) model_path = argv[++a];
        else if (!strcmp(argv[a], "--trace") && a + 1 < argc) trace = argv[++a];
        else if (!strcmp(argv[a], "--users") && a + 1 < argc) users = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--calls") && a + 1 < argc) calls = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--scale") && a + 1 < argc) scale = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--gpu-frac") && a + 1 < argc) gpu_frac = atof(argv[++a]);
        else if (!strcmp(argv[a], "--skip-bench")) bench = 0;
    }
    struct llama_model *model = ke_load_model(model_path);
    if (bench) microbench(model);

    trace_t tr = trace_load(trace);
    workload_t w = build_workload(&tr, users, 0);
    int ev0 = pick_window(&w, calls);
    int n_ev = calls < w.n_ev - ev0 ? calls : w.n_ev - ev0;

    /* 全部会话私有 KV 峰值之和，用来定各层容量 */
    char *seen = calloc(w.n_inst, 1);
    long peak = 0; int n_sess = 0, max_sess = 0;
    for (int k = 0; k < n_ev; k++) {
        int i = w.ev[ev0 + k].inst;
        if (seen[i]) continue;
        seen[i] = 1; n_sess++;
        const session_t *b = w.inst[i].base;
        int m = 0;
        for (int c = 0; c < b->n; c++) { int t = (b->calls[c].ctx - b->shared) / scale; if (t > m) m = t; }
        peak += m;
        if (m > max_sess) max_sess = m;
    }
    long gpu = (long)(peak * gpu_frac), dram;
    if (gpu < max_sess * 3 / 2) gpu = max_sess * 3 / 2;  /* 至少放得下最大的会话 */
    dram = 2 * gpu;
    int prefix_tok = 0;
    for (int p = 0; p < N_PREFIX; p++) prefix_tok += SHARED_PREFIXES[p] / scale;
    /* 物理 KV 容量：GPU 层预算 + 共享前缀 + 一个最大会话（不开 dedup 时含前缀）+ 余量 */
    int n_ctx = (int)(gpu + 2 * prefix_tok + max_sess + 4096);

    printf("\n== Part 2: replay of your real trace (%d users, %d calls, tokens / %d) ==\n", users, n_ev, scale);
    printf("%d sessions, private KV total %ld tokens; GPU tier %ld tokens, DRAM tier %ld tokens, SSD unlimited\n",
           n_sess, peak, gpu, dram);
    printf("%-22s %8s %7s %7s %7s %10s %9s %8s %8s %9s %8s %8s %9s\n", "policy", "GPU-hit", "DRAM", "SSD", "miss",
           "recompute", "meanTTFT", "p50", "p95", "bg I/O", "restore", "prefill", "pf tok/s");
    struct { int pol, dedup; const char *name; } runs[] = {
        {POL_LRU, 0, "LRU"}, {POL_LRU, 1, "LRU+shared"}, {POL_AGENT, 0, "signal-aware"},
        {POL_AGENT, 1, "signal-aware+shared"}};
    for (int r = 0; r < 4; r++) {
        summary_t x = replay(model, &w, ev0, n_ev, runs[r].pol, runs[r].dedup, scale, gpu, dram, n_ctx);
        printf("%-22s %7.1f%% %6.1f%% %6.1f%% %6.1f%% %10ld %8.0fms %6.0fms %6.0fms %8.1fs %6.0fms %6.0fms %9.0f\n", runs[r].name,
               100 * x.gpu, 100 * x.dram, 100 * x.ssd, 100 * x.miss, x.recompute, 1000 * x.mean,
               1000 * x.p50, 1000 * x.p95, x.io, 1000 * x.restore, 1000 * x.prefill, x.pf_tok);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
