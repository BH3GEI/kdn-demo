/* 生产规模的多租户回放：N 个"像你一样的用户"共享一台推理机。
 *
 * 三级存储 HBM / DRAM / SSD。每次请求到来时：
 *   KV 在 HBM：直接用；在 DRAM / SSD：按带宽搬回；都不在：按 prefill 速度重算。
 *
 * 用法：./sim [--kv mla|gqa|v41flash] [--hbm-gb 450] [--dram-gb 1000] [--ssd-tb 16] 50 100 200
 */
#include "common.h"

enum { HBM, DRAM, SSD, NTIER, NONE = -1 };

typedef struct {
    double cap, used;
    int *mem, n, capn; /* 成员列表，swap-remove */
} tier_t;

typedef struct {
    int tier, pos;            /* 所在层及其在成员列表中的下标 */
    double size;              /* 占用字节 */
    double end, gap, next;    /* 驱逐键所需 */
    predictor_t pred;
} istate_t;

static tier_t T[NTIER];
static istate_t *S;
static int POL;

static void tier_add(int t, int i, double size) {
    tier_t *x = &T[t];
    if (x->n == x->capn) { x->capn = x->capn ? x->capn * 2 : 256; x->mem = realloc(x->mem, x->capn * sizeof(int)); }
    S[i].tier = t; S[i].pos = x->n; S[i].size = size;
    x->mem[x->n++] = i; x->used += size;
}

static void tier_remove(int i) {
    tier_t *x = &T[S[i].tier];
    int last = x->mem[--x->n];
    x->mem[S[i].pos] = last; S[last].pos = S[i].pos;
    x->used -= S[i].size;
    S[i].tier = NONE;
}

static double key_of(int i, double now) {
    return evict_key(POL, now, S[i].end, S[i].gap, S[i].next);
}

/* 放进第 t 层；放不下就把该层最该走的往下挤，自己最该走就自己往下 */
static void put(int i, double size, int t, double now) {
    for (; t < NTIER; t++) {
        tier_t *x = &T[t];
        double mine = key_of(i, now);
        while (x->used + size > x->cap && x->n) {
            int v = -1; double vk = -INFINITY;
            for (int j = 0; j < x->n; j++) {
                double k = key_of(x->mem[j], now);
                if (k > vk) { vk = k; v = x->mem[j]; }
            }
            if (vk <= mine) break;
            double vs = S[v].size;
            tier_remove(v);
            put(v, vs, t + 1, now);
        }
        if (x->used + size <= x->cap) { tier_add(t, i, size); return; }
    }
}

typedef struct { double hbm, dram, ssd, miss, recompute, reuse, mean, p99; } result_t;

static int cmp_d(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return d < 0 ? -1 : d > 0;
}

static result_t simulate(const workload_t *w, int pol, double kvb, double hbm, double dram,
                         double ssd, int dedup) {
    const double BW_DRAM = 50e9, BW_SSD = 12e9, PREFILL_TPS = 15000;
    POL = pol;
    for (int t = 0; t < NTIER; t++) { T[t].used = 0; T[t].n = 0; }
    T[HBM].cap = hbm; T[DRAM].cap = dram; T[SSD].cap = ssd;
    if (dedup) for (int p = 0; p < N_PREFIX; p++) T[HBM].cap -= SHARED_PREFIXES[p] * kvb;
    S = realloc(S, w->n_inst * sizeof(istate_t));
    for (int i = 0; i < w->n_inst; i++) { S[i].tier = NONE; pred_init(&S[i].pred); }

    double cnt[NTIER + 1] = {0}, recompute = 0, reuse = 0;
    double *extra = malloc(w->n_ev * sizeof(double));
    int n = 0;
    for (int e = 0; e < w->n_ev; e++) {
        int i = w->ev[e].inst, c = w->ev[e].call;
        const session_t *base = w->inst[i].base;
        double shift = w->inst[i].shift;
        const call_t *cl = &base->calls[c];
        double start = cl->start + shift, end = cl->end + shift;
        int priv0 = dedup ? base->shared : 0;
        if (c > 0) {
            const call_t *p = &base->calls[c - 1];
            pred_observe(&S[i].pred, p->sig, start - (p->end + shift));
            double tok = fmax(p->ctx + p->out - priv0, 0), need = tok * kvb, cost;
            int where = S[i].tier;
            if (where != NONE) tier_remove(i);
            if (where == HBM) cost = 0;
            else if (where == DRAM) cost = need / BW_DRAM;
            else if (where == SSD) cost = need / BW_SSD;
            else { recompute += tok; cost = tok / PREFILL_TPS; }
            cnt[where == NONE ? NTIER : where] += 1;
            reuse += tok;
            extra[n++] = cost;
        }
        double next = c + 1 < base->n ? base->calls[c + 1].start + shift : end + 10 * DAY;
        double g = pred_gap(&S[i].pred, cl->sig);
        S[i].end = end; S[i].gap = g; S[i].next = next;
        int start_tier = (pol == POL_AGENT && g > COLD_GAP) ? SSD : HBM;
        put(i, fmax(cl->ctx + cl->out - priv0, 0) * kvb, start_tier, end);
    }
    double sum = 0;
    for (int k = 0; k < n; k++) sum += extra[k];
    qsort(extra, n, sizeof(double), cmp_d);
    result_t r = {cnt[HBM] / n, cnt[DRAM] / n, cnt[SSD] / n, cnt[NTIER] / n,
                  recompute / fmax(reuse, 1), 0, sum / n, extra[(int)(0.99 * n)]};
    free(extra);
    return r;
}

int main(int argc, char **argv) {
    double kvb = 70272, hbm = 450e9, dram = 1000e9, ssd = 16e12;
    const char *kvname = "mla", *path = "data/trace.csv";
    int users[32], nu = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--kv") && a + 1 < argc) {
            kvname = argv[++a];
            kvb = !strcmp(kvname, "gqa") ? 327680 : !strcmp(kvname, "v41flash") ? 890 : 70272;
        } else if (!strcmp(argv[a], "--hbm-gb") && a + 1 < argc) hbm = atof(argv[++a]) * 1e9;
        else if (!strcmp(argv[a], "--dram-gb") && a + 1 < argc) dram = atof(argv[++a]) * 1e9;
        else if (!strcmp(argv[a], "--ssd-tb") && a + 1 < argc) ssd = atof(argv[++a]) * 1e12;
        else if (!strcmp(argv[a], "--trace") && a + 1 < argc) path = argv[++a];
        else if (nu < 32) users[nu++] = atoi(argv[a]);
    }
    if (!nu) { users[0] = 50; users[1] = 100; users[2] = 200; nu = 3; }
    trace_t tr = trace_load(path);
    printf("KV=%s (%.0f B/token)  HBM %.0fGB  DRAM %.0fGB  SSD %.0fTB\n", kvname, kvb, hbm / 1e9, dram / 1e9, ssd / 1e12);
    printf("%5s  %-18s %8s %7s %7s %7s %10s %10s %9s\n", "users", "policy", "HBM-hit", "DRAM", "SSD",
           "miss", "recompute", "mean+TTFT", "p99+TTFT");
    struct { int pol, dedup; const char *name; } runs[] = {
        {POL_LRU, 0, "lru"}, {POL_LRU, 1, "lru+shared-prefix"},
        {POL_AGENT, 1, "agent+shared-prefix"}, {POL_ORACLE, 1, "oracle+shared-prefix"}};
    for (int u = 0; u < nu; u++) {
        workload_t w = build_workload(&tr, users[u], 0);
        for (int r = 0; r < 4; r++) {
            result_t x = simulate(&w, runs[r].pol, kvb, hbm, dram, ssd, runs[r].dedup);
            printf("%5d  %-18s %7.1f%% %6.1f%% %6.1f%% %6.2f%% %9.2f%% %9.2fs %8.1fs\n", users[u],
                   runs[r].name, 100 * x.hbm, 100 * x.dram, 100 * x.ssd, 100 * x.miss,
                   100 * x.recompute, x.mean, x.p99);
        }
        printf("\n");
        free(w.inst); free(w.ev);
    }
    return 0;
}
