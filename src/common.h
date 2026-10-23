/* 轨迹读取、多租户叠加、等待信号与驱逐策略。sim.c 和 live.c 共用。 */
#ifndef KDN_COMMON_H
#define KDN_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 调用结束时的等待信号 ---------- */
enum { SIG_LOCAL, SIG_MCP, SIG_AGENT, SIG_MONITOR, SIG_WAKEUP, SIG_BASH, SIG_WEB,
       SIG_OTHER, SIG_HUMAN, SIG_ASK, SIG_END, N_SIG };

/* 各信号的默认预测空档（秒）；会话内有历史时用上一次同类信号的实际空档 */
static const double DEFAULT_GAP[N_SIG] = {0.5, 1, 1, 1, 1, 3, 20, 5, 60, 120, 3600};
static const int USE_HISTORY[N_SIG] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0};
#define COLD_GAP (30 * 60.0) /* 预测空档超过 30 分钟：直接降到 SSD */

enum { POL_LRU, POL_AGENT, POL_ORACLE };
static const char *POL_NAME[] = {"lru", "agent", "oracle"};

typedef struct {
    double last[N_SIG]; /* 上一次同类信号的实际空档，<0 表示没有 */
} predictor_t;

static inline void pred_init(predictor_t *p) {
    for (int i = 0; i < N_SIG; i++) p->last[i] = -1;
}
static inline double pred_gap(const predictor_t *p, int sig) {
    return (USE_HISTORY[sig] && p->last[sig] >= 0) ? p->last[sig] : DEFAULT_GAP[sig];
}
static inline void pred_observe(predictor_t *p, int sig, double gap) {
    if (USE_HISTORY[sig]) p->last[sig] = gap;
}

/* 驱逐优先级：越大越先挪走。end = 上次调用结束，gap = 预测空档，next = 真实下次使用 */
static inline double evict_key(int pol, double now, double end, double gap, double next) {
    if (pol == POL_LRU) return now - end;
    if (pol == POL_ORACLE) return next - now;
    double remaining = end + gap - now;
    /* 超时未到：已等待 w 秒的重尾等待，期望剩余时间约等于 w */
    return remaining >= 0 ? remaining : now - end;
}

/* ---------- 轨迹 ---------- */
typedef struct { double start, end; int ctx, out, sig; } call_t;
typedef struct { int shared, n, cap; call_t *calls; } session_t;
typedef struct { int n; session_t *s; } trace_t;

static trace_t trace_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    trace_t t = {0, NULL};
    int cap = 0;
    char line[256];
    fgets(line, sizeof line, f); /* 表头 */
    int sid, shared, ctx, out, sig;
    double a, b;
    while (fscanf(f, "%d,%d,%lf,%lf,%d,%d,%d\n", &sid, &shared, &a, &b, &ctx, &out, &sig) == 7) {
        if (sid >= cap) {
            int nc = cap ? cap * 2 : 1024;
            while (nc <= sid) nc *= 2;
            t.s = realloc(t.s, nc * sizeof(session_t));
            memset(t.s + cap, 0, (nc - cap) * sizeof(session_t));
            cap = nc;
        }
        if (sid >= t.n) t.n = sid + 1;
        session_t *s = &t.s[sid];
        s->shared = shared;
        if (s->n == s->cap) {
            s->cap = s->cap ? s->cap * 2 : 16;
            s->calls = realloc(s->calls, s->cap * sizeof(call_t));
        }
        s->calls[s->n++] = (call_t){a, b, ctx, out, sig};
    }
    fclose(f);
    return t;
}

/* ---------- 多租户叠加 ----------
 * 用户 u 用的是你第 days[u % D] 天的真实轨迹，叠到同一天（保留一天内的时刻）；
 * 同一天被重复使用时再随机错开 0–60 分钟。 */
#define DAY 86400.0
#define TZ (8 * 3600.0)

typedef struct { const session_t *base; double shift; } inst_t;   /* 某用户的一个会话 */
typedef struct { double t; int inst, call; } event_t;
typedef struct { int n_inst, n_ev; inst_t *inst; event_t *ev; } workload_t;

static uint64_t rng_state = 88172645463325252ull;
static inline uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double rng_uniform(void) { return (rng_next() >> 11) * (1.0 / 9007199254740992.0); }

static int cmp_event(const void *a, const void *b) {
    double d = ((const event_t *)a)->t - ((const event_t *)b)->t;
    return d < 0 ? -1 : d > 0;
}

static workload_t build_workload(const trace_t *tr, int n_users, uint64_t seed) {
    rng_state = seed * 2654435761ull + 88172645463325252ull;
    /* 按天分组，只要调用数 >= 200 的天 */
    int dmin = 1 << 30, dmax = 0;
    int *sday = malloc(tr->n * sizeof(int));
    for (int i = 0; i < tr->n; i++) {
        sday[i] = tr->s[i].n ? (int)floor((tr->s[i].calls[0].start + TZ) / DAY) : -1;
        if (sday[i] < 0) continue;
        if (sday[i] < dmin) dmin = sday[i];
        if (sday[i] > dmax) dmax = sday[i];
    }
    int nd = dmax - dmin + 1, *cnt = calloc(nd, sizeof(int)), *days = malloc(nd * sizeof(int)), D = 0;
    for (int i = 0; i < tr->n; i++) if (sday[i] >= 0) cnt[sday[i] - dmin] += tr->s[i].n;
    for (int d = 0; d < nd; d++) if (cnt[d] >= 200) days[D++] = d + dmin;
    for (int i = D - 1; i > 0; i--) { int j = rng_next() % (i + 1), x = days[i]; days[i] = days[j]; days[j] = x; }

    workload_t w = {0, 0, NULL, NULL};
    int icap = 0, ecap = 0;
    for (int u = 0; u < n_users; u++) {
        int d = days[u % D];
        double shift = -d * DAY + (u >= D ? rng_uniform() * 3600 : 0);
        for (int i = 0; i < tr->n; i++) {
            if (sday[i] != d) continue;
            if (w.n_inst == icap) { icap = icap ? icap * 2 : 1024; w.inst = realloc(w.inst, icap * sizeof(inst_t)); }
            w.inst[w.n_inst] = (inst_t){&tr->s[i], shift};
            for (int c = 0; c < tr->s[i].n; c++) {
                if (w.n_ev == ecap) { ecap = ecap ? ecap * 2 : 65536; w.ev = realloc(w.ev, ecap * sizeof(event_t)); }
                w.ev[w.n_ev++] = (event_t){tr->s[i].calls[c].start + shift, w.n_inst, c};
            }
            w.n_inst++;
        }
    }
    qsort(w.ev, w.n_ev, sizeof(event_t), cmp_event);
    free(sday); free(cnt); free(days);
    return w;
}

/* 共享前缀（系统提示 + 工具定义）各组一份 */
static const int SHARED_PREFIXES[] = {60000, 16836, 35000};
#define N_PREFIX 3
static inline int prefix_index(int shared) {
    for (int i = 0; i < N_PREFIX; i++) if (SHARED_PREFIXES[i] == shared) return i;
    return 0;
}

#endif
