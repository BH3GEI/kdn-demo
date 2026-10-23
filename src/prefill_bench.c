/* prefill 速度排查。
 *
 * 第一部分：从空 cache 开始 prefill，按深度分段测 tok/s，比较上下文参数
 *          （flash attention、ubatch、n_seq_max、kv_unified）。
 * 第二部分：kv_unified 下，别的序列占着 N 个 KV 单元时，
 *          一个新请求的 prefill 会不会变慢。
 *
 * 用法：./prefill_bench [--model models/qwen3-1.7b-q4_k_m.gguf] [--len 16384]
 */
#include "common.h"
#include "llama.h"

#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

typedef struct { const char *name; int fa, ubatch, n_seq, unified; } cfg_t;

static struct llama_context *make_ctx(struct llama_model *m, const cfg_t *c, int n_ctx) {
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = c->ubatch;
    cp.n_seq_max = c->n_seq;
    cp.kv_unified = c->unified;
    cp.no_perf = true;
    cp.flash_attn_type = c->fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "failed to create context (%s)\n", c->name); exit(1); }
    return ctx;
}

/* 在 seq 上从 pos0 开始 prefill n 个 token；seg>0 时每 seg 个 token 记一次耗时 */
static double prefill(struct llama_context *ctx, struct llama_batch *b, int bsz, int seq, int pos0, int n,
                      int seg, double *seg_t) {
    double t0 = now(), ts = t0;
    for (int i = 0; i < n; i += bsz) {
        int m = n - i < bsz ? n - i : bsz;
        b->n_tokens = m;
        for (int j = 0; j < m; j++) {
            b->token[j] = 1000 + rng_next() % 149000;
            b->pos[j] = pos0 + i + j;
            b->n_seq_id[j] = 1;
            b->seq_id[j][0] = seq;
            b->logits[j] = (i + j == n - 1);
        }
        if (llama_decode(ctx, *b)) { fprintf(stderr, "decode failed\n"); exit(1); }
        if (seg && (i + m) % seg == 0) {
            llama_synchronize(ctx);
            double t = now();
            seg_t[(i + m) / seg - 1] = t - ts;
            ts = t;
        }
    }
    llama_synchronize(ctx);
    return now() - t0;
}

int main(int argc, char **argv) {
    const char *path = "models/qwen3-1.7b-q4_k_m.gguf";
    int len = 16384, part1 = 1;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--model") && a + 1 < argc) path = argv[++a];
        else if (!strcmp(argv[a], "--len") && a + 1 < argc) len = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--skip-part1")) part1 = 0;
    }
    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++)
        printf("device %zu: %s\n", i, ggml_backend_dev_description(ggml_backend_dev_get(i)));
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) { fprintf(stderr, "failed to load %s\n", path); return 1; }

    /* ---------- 第一部分 ---------- */
    const int SEG = 2048, nseg = len / SEG;
    static const cfg_t cfgs[] = {
        {"live.c settings", 1, 1024, 128, 1},
        {"single seq", 1, 1024, 1, 0},
        {"single seq, ubatch 512", 1, 512, 1, 0},
        {"single seq, ubatch 2048", 1, 2048, 1, 0},
        {"single seq, no FA", 0, 1024, 1, 0},
    };
    if (part1) {
    printf("\n== Part 1: prefill %d tokens from empty, tok/s per %d-token segment ==\n", len, SEG);
    printf("%-26s %8s", "config", "overall");
    for (int s = 0; s < nseg; s++) printf(" %6dk", (s + 1) * SEG / 1024);
    printf("\n");
    double *seg_t = malloc(nseg * sizeof(double));
    for (size_t c = 0; c < sizeof cfgs / sizeof cfgs[0]; c++) {
        struct llama_context *ctx = make_ctx(m, &cfgs[c], len + 256);
        struct llama_batch b = llama_batch_init(cfgs[c].ubatch, 0, 1);
        prefill(ctx, &b, cfgs[c].ubatch, 0, 0, 512, 0, NULL); /* 预热 */
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        double t = prefill(ctx, &b, cfgs[c].ubatch, 0, 0, len, SEG, seg_t);
        printf("%-26s %8.0f", cfgs[c].name, len / t);
        for (int s = 0; s < nseg; s++) printf(" %7.0f", SEG / seg_t[s]);
        printf("\n");
        fflush(stdout);
        llama_batch_free(b);
        llama_free(ctx);
    }
    }

    /* ---------- 第二部分 ---------- */
    static const int OCC[] = {0, 8192, 32768, 65536};
    static const int REQS[] = {64, 128, 512};
    const int REQ_CTX = 2048, REPS = 5;
    printf("\n== Part 2: kv_unified, other sequences hold N cells; request on a %d-token session, prefill ms (tok/s) ==\n", REQ_CTX);
    printf("%10s", "occupied");
    for (int q = 0; q < 3; q++) printf("   %4d new tok", REQS[q]);
    printf("\n");
    const cfg_t *u = &cfgs[0];
    struct llama_context *ctx = make_ctx(m, u, 65536 + REQ_CTX + 3 * 512 * REPS + 1024);
    llama_memory_t mem = llama_get_memory(ctx);
    struct llama_batch b = llama_batch_init(u->ubatch, 0, 1);
    for (size_t k = 0; k < sizeof OCC / sizeof OCC[0]; k++) {
        llama_memory_clear(mem, true);
        /* 用 8k 一个的"别的会话"占住单元 */
        for (int off = 0, s = 2; off < OCC[k]; off += 8192, s++) prefill(ctx, &b, u->ubatch, s, 0, 8192, 0, NULL);
        prefill(ctx, &b, u->ubatch, 1, 0, REQ_CTX, 0, NULL);
        int pos = REQ_CTX;
        printf("%10d", OCC[k]);
        for (int q = 0; q < 3; q++) {
            double tot = 0;
            for (int r = 0; r < REPS; r++) {
                double t = prefill(ctx, &b, u->ubatch, 1, pos, REQS[q], 0, NULL);
                pos += REQS[q];
                if (r) tot += t;  /* 第一次当预热 */
            }
            double ms = 1000 * tot / (REPS - 1);
            printf("   %5.1f (%4.0f)", ms, REQS[q] / ms * 1000);
        }
        printf("\n");
        fflush(stdout);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
