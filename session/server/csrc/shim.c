/* llama.cpp 的薄 C 垫片，给 Rust 用。
 *
 * 只暴露基本类型参数（int / 指针），Rust 侧不用镜像 llama_context_params、llama_batch
 * 这些随版本变化的结构体布局 —— 结构体在这里按头文件编译，ABI 永远对得上。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

typedef struct {
    struct llama_context *ctx;
    llama_memory_t mem;
    struct llama_batch batch;
    int n_batch;
} kdn_ctx;

static void quiet_log(enum ggml_log_level level, const char *text, void *ud) {
    (void)ud;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static bool no_progress(float p, void *ud) { (void)p; (void)ud; return true; }

struct llama_model *kdn_load_model(const char *path, int n_gpu_layers) {
    llama_log_set(quiet_log, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.progress_callback = no_progress;
    return llama_model_load_from_file(path, mp);
}

void kdn_free_model(struct llama_model *m) { llama_model_free(m); }

kdn_ctx *kdn_new_ctx(struct llama_model *model, int n_ctx, int n_batch, int n_seq_max) {
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = n_batch;
    cp.n_seq_max = n_seq_max;
    cp.kv_unified = true; /* 序列之间共享 KV 单元：seq_cp（fork、共享前缀）才是真共享 */
    cp.no_perf = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return NULL;
    kdn_ctx *k = calloc(1, sizeof(kdn_ctx));
    k->ctx = ctx;
    k->mem = llama_get_memory(ctx);
    k->batch = llama_batch_init(n_batch, 0, 1);
    k->n_batch = n_batch;
    return k;
}

void kdn_free_ctx(kdn_ctx *k) {
    llama_batch_free(k->batch);
    llama_free(k->ctx);
    free(k);
}

/* 在序列 seq 的 pos0 处依次喂入 n 个 token，按 n_batch 分块；want_logits 时最后一个出 logits。
 * 返回 0 成功，否则为 llama_decode 的返回码。 */
int kdn_decode(kdn_ctx *k, const int32_t *toks, int n, int pos0, int seq, int want_logits) {
    for (int i = 0; i < n; i += k->n_batch) {
        int m = n - i < k->n_batch ? n - i : k->n_batch;
        k->batch.n_tokens = m;
        for (int j = 0; j < m; j++) {
            k->batch.token[j] = toks[i + j];
            k->batch.pos[j] = pos0 + i + j;
            k->batch.n_seq_id[j] = 1;
            k->batch.seq_id[j][0] = seq;
            k->batch.logits[j] = want_logits && (i + j == n - 1);
        }
        int rc = llama_decode(k->ctx, k->batch);
        if (rc) return rc;
    }
    return 0;
}

const float *kdn_logits_last(kdn_ctx *k) { return llama_get_logits_ith(k->ctx, -1); }

/* ---------- 序列 / 状态 ---------- */
int kdn_seq_rm(kdn_ctx *k, int seq, int p0, int p1) { return llama_memory_seq_rm(k->mem, seq, p0, p1); }
void kdn_seq_cp(kdn_ctx *k, int src, int dst, int p0, int p1) { llama_memory_seq_cp(k->mem, src, dst, p0, p1); }
int kdn_seq_pos_max(kdn_ctx *k, int seq) { return llama_memory_seq_pos_max(k->mem, seq); }
size_t kdn_state_seq_size(kdn_ctx *k, int seq) { return llama_state_seq_get_size(k->ctx, seq); }
size_t kdn_state_seq_get(kdn_ctx *k, uint8_t *dst, size_t n, int seq) { return llama_state_seq_get_data(k->ctx, dst, n, seq); }
size_t kdn_state_seq_set(kdn_ctx *k, const uint8_t *src, size_t n, int seq) { return llama_state_seq_set_data(k->ctx, src, n, seq); }

/* ---------- 词表 ---------- */
int kdn_n_vocab(const struct llama_model *m) { return llama_vocab_n_tokens(llama_model_get_vocab(m)); }
int kdn_add_bos(const struct llama_model *m) { return llama_vocab_get_add_bos(llama_model_get_vocab(m)); }
int kdn_is_eog(const struct llama_model *m, int32_t t) { return llama_vocab_is_eog(llama_model_get_vocab(m), t); }

int kdn_tokenize(const struct llama_model *m, const char *text, int len, int32_t *out, int max, int add_special) {
    /* parse_special 恒为真：模板里的 <|im_start|>、<tool_call> 等要按特殊 token 切分，和 HF 一致 */
    return llama_tokenize(llama_model_get_vocab(m), text, len, out, max, add_special, true);
}

int kdn_token_to_piece(const struct llama_model *m, int32_t t, char *buf, int len) {
    return llama_token_to_piece(llama_model_get_vocab(m), t, buf, len, 0, true);
}

int kdn_meta_str(const struct llama_model *m, const char *key, char *buf, size_t len) {
    return llama_model_meta_val_str(m, key, buf, len);
}

/* llama_decode 在 Metal 上是异步的；计时前先等 GPU 做完 */
void kdn_sync(kdn_ctx *k) { llama_synchronize(k->ctx); }
