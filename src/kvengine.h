/* MacBook 上真跑的分层 KV 推理引擎（llama.cpp C API，Metal）。
 *
 * 三层：
 *   gpu  ：llama.cpp KV cache 里的一个序列（slot），直接接着算
 *   dram ：llama_state_seq_get_data 拷到主机缓冲区
 *   ssd  ：缓冲区写成文件（F_NOCACHE，绕过页缓存，读写都真打到盘上）
 * 都不在就从头 prefill。
 *
 * 共享前缀（系统提示 + 工具定义）各算一次，放在常驻序列 0..N_PREFIX-1；
 * 会话用 llama_memory_seq_cp 引用这些 KV 单元（真实共享，不复制）。
 * 开启 dedup 时，降级只存私有部分，恢复后再把共享前缀挂回来。
 */
#ifndef KDN_KVENGINE_H
#define KDN_KVENGINE_H

#include "common.h"
#include "llama.h"

enum { T_GPU, T_DRAM, T_SSD, T_NONE = -1 };

typedef struct {
    int id, prefix;
    llama_token *hist; int nhist, caphist; /* 共享前缀之后的全部 token（重算用） */
    int tier, slot;
    uint8_t *buf; size_t bufsz;            /* dram 层 */
    char path[256];                        /* ssd 层 */
    double end, gap, next;                 /* 驱逐键 */
    predictor_t pred;
} ksess_t;

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    llama_memory_t mem;
    struct llama_batch batch;
    int n_batch, n_seq_max;
    int policy, dedup;
    long cap[3], used[3];                  /* 各层容量 / 占用（token） */
    int plen[N_PREFIX];
    int *slot_free; int n_free;
    ksess_t **sess; int n_sess, cap_sess;
    char ssd_dir[200];
    double io_s;                           /* 后台降级 I/O 时间（不计入 TTFT） */
} kengine_t;

typedef struct {
    int where;          /* 请求到来时 KV 在哪一层；-2 表示新会话 */
    double restore_s, ttft_s;
    int recompute_tok, new_tok;
} kresult_t;

struct llama_model *ke_load_model(const char *path);
kengine_t *ke_new(struct llama_model *model, int n_ctx, long gpu_tok, long dram_tok,
                  int policy, int dedup, const char *ssd_dir);
void ke_free(kengine_t *e);
void ke_register_prefix(kengine_t *e, int idx, const llama_token *toks, int n);
kresult_t ke_request(kengine_t *e, int sid, int prefix, const llama_token *toks, int n, double now);
void ke_finish(kengine_t *e, int sid, int sig, double now, double true_next);
void ke_observe(kengine_t *e, int sid, int sig, double gap);

/* 微基准用 */
double ke_now(void);
int ke_prefill(kengine_t *e, int seq, int pos0, const llama_token *toks, int n);
ksess_t *ke_get(kengine_t *e, int sid, int prefix);
void ke_demote(kengine_t *e, ksess_t *s, int to, double now);
double ke_restore(kengine_t *e, ksess_t *s);
int ke_alloc_slot(kengine_t *e, double now);

#endif
