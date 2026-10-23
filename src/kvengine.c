#include "kvengine.h"

#include <fcntl.h>
#include <time.h>
#include <unistd.h>

double ke_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

struct llama_model *ke_load_model(const char *path) {
    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) { fprintf(stderr, "failed to load %s\n", path); exit(1); }
    return m;
}

kengine_t *ke_new(struct llama_model *model, int n_ctx, long gpu_tok, long dram_tok,
                  int policy, int dedup, const char *ssd_dir) {
    kengine_t *e = calloc(1, sizeof(kengine_t));
    e->model = model;
    e->vocab = llama_model_get_vocab(model);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = e->n_batch = 1024;
    cp.n_seq_max = e->n_seq_max = 128;
    cp.kv_unified = true;  /* 序列之间共享 KV 单元，共享前缀才能不复制 */
    cp.no_perf = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    e->ctx = llama_init_from_model(model, cp);
    if (!e->ctx) { fprintf(stderr, "failed to create context\n"); exit(1); }
    e->mem = llama_get_memory(e->ctx);
    e->batch = llama_batch_init(e->n_batch, 0, 1);
    e->policy = policy; e->dedup = dedup;
    e->cap[T_GPU] = gpu_tok; e->cap[T_DRAM] = dram_tok; e->cap[T_SSD] = 1L << 60;
    e->slot_free = malloc(e->n_seq_max * sizeof(int));
    for (int s = e->n_seq_max - 1; s >= N_PREFIX; s--) e->slot_free[e->n_free++] = s;
    snprintf(e->ssd_dir, sizeof e->ssd_dir, "%s", ssd_dir);
    char cmd[256];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", ssd_dir);
    system(cmd);
    return e;
}

void ke_free(kengine_t *e) {
    for (int i = 0; i < e->n_sess; i++) {
        ksess_t *s = e->sess[i];
        if (!s) continue;
        if (s->tier == T_SSD) unlink(s->path);
        free(s->buf); free(s->hist); free(s);
    }
    free(e->sess); free(e->slot_free);
    llama_batch_free(e->batch);
    llama_free(e->ctx);
    free(e);
}

/* 在序列 seq 的 pos0 处 prefill，最后一个 token 出 logits；返回贪心采样的首 token */
int ke_prefill(kengine_t *e, int seq, int pos0, const llama_token *toks, int n) {
    int out = -1;
    for (int i = 0; i < n; i += e->n_batch) {
        int m = n - i < e->n_batch ? n - i : e->n_batch;
        e->batch.n_tokens = m;
        for (int j = 0; j < m; j++) {
            e->batch.token[j] = toks[i + j];
            e->batch.pos[j] = pos0 + i + j;
            e->batch.n_seq_id[j] = 1;
            e->batch.seq_id[j][0] = seq;
            e->batch.logits[j] = (i + j == n - 1);
        }
        if (llama_decode(e->ctx, e->batch)) { fprintf(stderr, "llama_decode failed (seq %d)\n", seq); exit(1); }
    }
    if (n > 0) {
        const float *lg = llama_get_logits_ith(e->ctx, -1);
        int nv = llama_vocab_n_tokens(e->vocab);
        out = 0;
        for (int v = 1; v < nv; v++) if (lg[v] > lg[out]) out = v;
    }
    return out;
}

void ke_register_prefix(kengine_t *e, int idx, const llama_token *toks, int n) {
    ke_prefill(e, idx, 0, toks, n);
    e->plen[idx] = n;
}

ksess_t *ke_get(kengine_t *e, int sid, int prefix) {
    if (sid >= e->cap_sess) {
        int nc = e->cap_sess ? e->cap_sess * 2 : 256;
        while (nc <= sid) nc *= 2;
        e->sess = realloc(e->sess, nc * sizeof(ksess_t *));
        memset(e->sess + e->cap_sess, 0, (nc - e->cap_sess) * sizeof(ksess_t *));
        e->cap_sess = nc;
    }
    if (sid >= e->n_sess) e->n_sess = sid + 1;
    if (!e->sess[sid]) {
        ksess_t *s = calloc(1, sizeof(ksess_t));
        s->id = sid; s->prefix = prefix; s->tier = T_NONE; s->slot = -1;
        pred_init(&s->pred);
        e->sess[sid] = s;
    }
    return e->sess[sid];
}

/* 占用（token）：开 dedup 时只算私有部分 */
static long footprint(kengine_t *e, ksess_t *s) {
    return s->nhist + (e->dedup ? 0 : e->plen[s->prefix]);
}

static double key_of(kengine_t *e, ksess_t *s, double now) {
    return evict_key(e->policy, now, s->end, s->gap, s->next);
}

static ksess_t *pick_victim(kengine_t *e, int tier, ksess_t *except, double now) {
    ksess_t *v = NULL; double vk = -INFINITY;
    for (int i = 0; i < e->n_sess; i++) {
        ksess_t *s = e->sess[i];
        if (!s || s == except || s->tier != tier) continue;
        double k = key_of(e, s, now);
        if (k > vk) { vk = k; v = s; }
    }
    return v;
}

/* ---------- 层间搬运 ---------- */
#define CHUNK (1u << 30) /* macOS 单次 read/write 不能超过 INT_MAX */
static void write_file(const char *path, const uint8_t *buf, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    fcntl(fd, F_NOCACHE, 1);
    for (size_t off = 0; off < n;) { ssize_t w = write(fd, buf + off, n - off > CHUNK ? CHUNK : n - off); if (w <= 0) break; off += w; }
    fsync(fd);
    close(fd);
}

static uint8_t *read_file(const char *path, size_t n) {
    uint8_t *buf = malloc(n);
    int fd = open(path, O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);  /* 绕过页缓存：真实的读盘 */
    for (size_t off = 0; off < n;) { ssize_t r = read(fd, buf + off, n - off > CHUNK ? CHUNK : n - off); if (r <= 0) break; off += r; }
    close(fd);
    return buf;
}

static void gpu_to_host(kengine_t *e, ksess_t *s) {
    if (e->dedup) llama_memory_seq_rm(e->mem, s->slot, 0, e->plen[s->prefix]); /* 只留私有部分 */
    s->bufsz = llama_state_seq_get_size(e->ctx, s->slot);
    s->buf = malloc(s->bufsz);
    llama_state_seq_get_data(e->ctx, s->buf, s->bufsz, s->slot);
    llama_memory_seq_rm(e->mem, s->slot, -1, -1);
    e->slot_free[e->n_free++] = s->slot;
    s->slot = -1;
}

void ke_demote(kengine_t *e, ksess_t *s, int to, double now);

static int make_room(kengine_t *e, int tier, long size, double now, ksess_t *incoming) {
    while (e->used[tier] + size > e->cap[tier]) {
        ksess_t *v = pick_victim(e, tier, incoming, now);
        if (!v) return 0;
        if (incoming && key_of(e, v, now) <= key_of(e, incoming, now)) return 0; /* 留下的都比我更早要用 */
        if (tier == T_GPU) ke_demote(e, v, T_DRAM, now);
        else if (tier == T_DRAM) ke_demote(e, v, T_SSD, now);
        else return 0;
    }
    return 1;
}

void ke_demote(kengine_t *e, ksess_t *s, int to, double now) {
    double t0 = ke_now();
    long size = footprint(e, s);
    e->used[s->tier] -= size;
    if (s->tier == T_GPU) gpu_to_host(e, s);
    if (to == T_DRAM && !make_room(e, T_DRAM, size, now, s)) to = T_SSD;
    if (to == T_SSD) {
        snprintf(s->path, sizeof s->path, "%s/s%d.kv", e->ssd_dir, s->id);
        write_file(s->path, s->buf, s->bufsz);
        free(s->buf); s->buf = NULL;
    }
    s->tier = to;
    e->used[to] += size;
    e->io_s += ke_now() - t0;
}

int ke_alloc_slot(kengine_t *e, double now) {
    while (!e->n_free) {
        ksess_t *v = pick_victim(e, T_GPU, NULL, now);
        ke_demote(e, v, T_DRAM, now);
    }
    return e->slot_free[--e->n_free];
}

/* 把会话 KV 搬回 gpu，返回搬运耗时 */
double ke_restore(kengine_t *e, ksess_t *s) {
    double t0 = ke_now();
    if (s->tier == T_SSD) {
        s->buf = read_file(s->path, s->bufsz);
        unlink(s->path);
    }
    s->slot = ke_alloc_slot(e, s->end);
    if (!llama_state_seq_set_data(e->ctx, s->buf, s->bufsz, s->slot)) {
        fprintf(stderr, "restore failed (session %d)\n", s->id); exit(1);
    }
    if (e->dedup) llama_memory_seq_cp(e->mem, s->prefix, s->slot, -1, -1); /* 挂回共享前缀 */
    free(s->buf); s->buf = NULL;
    s->tier = T_GPU;
    return ke_now() - t0;
}

/* ---------- 对外接口 ---------- */
kresult_t ke_request(kengine_t *e, int sid, int prefix, const llama_token *toks, int n, double now) {
    ksess_t *s = ke_get(e, sid, prefix);
    kresult_t r = {s->tier, 0, 0, 0, 0, n};
    if (s->tier == T_NONE && s->nhist == 0) r.where = -2;
    double t0 = ke_now();
    if (s->tier != T_NONE) e->used[s->tier] -= footprint(e, s);
    if (s->tier == T_DRAM || s->tier == T_SSD) r.restore_s = ke_restore(e, s);
    if (s->tier == T_NONE) {
        /* 冷启动或已被丢弃：共享前缀直接引用，私有部分重算 */
        s->slot = ke_alloc_slot(e, now);
        llama_memory_seq_cp(e->mem, prefix, s->slot, -1, -1);
        ke_prefill(e, s->slot, e->plen[prefix], s->hist, s->nhist);
        r.recompute_tok = r.where == -2 ? 0 : s->nhist;
    }
    double tp = ke_now();
    ke_prefill(e, s->slot, e->plen[prefix] + s->nhist, toks, n);
    r.prefill_s = ke_now() - tp;
    r.ttft_s = ke_now() - t0;
    if (s->nhist + n > s->caphist) {
        while (s->nhist + n > s->caphist) s->caphist = s->caphist ? s->caphist * 2 : 1024;
        s->hist = realloc(s->hist, s->caphist * sizeof(llama_token));
    }
    memcpy(s->hist + s->nhist, toks, n * sizeof(llama_token));
    s->nhist += n;
    s->tier = T_GPU;
    return r;
}

void ke_finish(kengine_t *e, int sid, int sig, double now, double true_next) {
    ksess_t *s = e->sess[sid];
    s->gap = pred_gap(&s->pred, sig);
    s->end = now; s->next = true_next;
    long size = footprint(e, s);
    int cold = e->policy == POL_AGENT && s->gap > COLD_GAP;
    int fits = !cold && make_room(e, T_GPU, size, now, s);
    e->used[T_GPU] += size;
    if (!fits) ke_demote(e, s, cold ? T_SSD : T_DRAM, now);
}

void ke_observe(kengine_t *e, int sid, int sig, double gap) {
    if (sid < e->n_sess && e->sess[sid]) pred_observe(&e->sess[sid]->pred, sig, gap);
}
