//! llama.cpp 的安全封装（经由 csrc/shim.c）。
//!
//! 只在引擎线程里使用：Model / Ctx 都不是 Send 安全的语义，这里也不实现 Send。

use std::ffi::{c_char, c_int, CString};

#[repr(C)]
pub struct RawModel {
    _p: [u8; 0],
}
#[repr(C)]
pub struct RawCtx {
    _p: [u8; 0],
}

extern "C" {
    fn kdn_load_model(path: *const c_char, n_gpu_layers: c_int) -> *mut RawModel;
    fn kdn_free_model(m: *mut RawModel);
    fn kdn_new_ctx(m: *mut RawModel, n_ctx: c_int, n_batch: c_int, n_seq_max: c_int) -> *mut RawCtx;
    fn kdn_free_ctx(k: *mut RawCtx);
    fn kdn_decode(k: *mut RawCtx, toks: *const i32, n: c_int, pos0: c_int, seq: c_int, want_logits: c_int) -> c_int;
    fn kdn_sync(k: *mut RawCtx);
    fn kdn_logits_last(k: *mut RawCtx) -> *const f32;
    fn kdn_seq_rm(k: *mut RawCtx, seq: c_int, p0: c_int, p1: c_int) -> c_int;
    fn kdn_seq_cp(k: *mut RawCtx, src: c_int, dst: c_int, p0: c_int, p1: c_int);
    fn kdn_seq_pos_max(k: *mut RawCtx, seq: c_int) -> c_int;
    fn kdn_state_seq_size(k: *mut RawCtx, seq: c_int) -> usize;
    fn kdn_state_seq_get(k: *mut RawCtx, dst: *mut u8, n: usize, seq: c_int) -> usize;
    fn kdn_state_seq_set(k: *mut RawCtx, src: *const u8, n: usize, seq: c_int) -> usize;
    fn kdn_n_vocab(m: *const RawModel) -> c_int;
    fn kdn_add_bos(m: *const RawModel) -> c_int;
    fn kdn_is_eog(m: *const RawModel, t: i32) -> c_int;
    fn kdn_tokenize(m: *const RawModel, text: *const c_char, len: c_int, out: *mut i32, max: c_int, add_special: c_int) -> c_int;
    fn kdn_token_to_piece(m: *const RawModel, t: i32, buf: *mut c_char, len: c_int) -> c_int;
    fn kdn_meta_str(m: *const RawModel, key: *const c_char, buf: *mut c_char, len: usize) -> c_int;
}

pub struct Model {
    raw: *mut RawModel,
    pub n_vocab: usize,
}

impl Model {
    pub fn load(path: &str, n_gpu_layers: i32) -> anyhow::Result<Model> {
        let c = CString::new(path)?;
        let raw = unsafe { kdn_load_model(c.as_ptr(), n_gpu_layers) };
        anyhow::ensure!(!raw.is_null(), "加载模型失败: {path}");
        let n_vocab = unsafe { kdn_n_vocab(raw) } as usize;
        Ok(Model { raw, n_vocab })
    }

    pub fn add_bos(&self) -> bool {
        unsafe { kdn_add_bos(self.raw) != 0 }
    }

    pub fn is_eog(&self, t: i32) -> bool {
        unsafe { kdn_is_eog(self.raw, t) != 0 }
    }

    /// 分词；特殊 token（<|im_start|> 等）按特殊 token 解析。
    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<i32> {
        let mut out = vec![0i32; text.len() + 8];
        let n = unsafe {
            kdn_tokenize(self.raw, text.as_ptr() as *const c_char, text.len() as c_int, out.as_mut_ptr(), out.len() as c_int, add_special as c_int)
        };
        assert!(n >= 0, "tokenize 缓冲区不够");
        out.truncate(n as usize);
        out
    }

    /// token -> 原始字节（可能是半个 UTF-8 字符）
    pub fn piece(&self, t: i32) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        let mut n = unsafe { kdn_token_to_piece(self.raw, t, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe { kdn_token_to_piece(self.raw, t, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        }
        buf.truncate(n.max(0) as usize);
        buf
    }

    pub fn detokenize(&self, toks: &[i32]) -> String {
        let bytes: Vec<u8> = toks.iter().flat_map(|&t| self.piece(t)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn meta(&self, key: &str) -> Option<String> {
        let k = CString::new(key).ok()?;
        let n = unsafe { kdn_meta_str(self.raw, k.as_ptr(), std::ptr::null_mut(), 0) };
        if n < 0 {
            return None;
        }
        let mut buf = vec![0u8; n as usize + 1];
        unsafe { kdn_meta_str(self.raw, k.as_ptr(), buf.as_mut_ptr() as *mut c_char, buf.len()) };
        buf.truncate(n as usize);
        String::from_utf8(buf).ok()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { kdn_free_model(self.raw) }
    }
}

pub struct Ctx {
    raw: *mut RawCtx,
    n_vocab: usize,
}

impl Ctx {
    pub fn new(model: &Model, n_ctx: usize, n_batch: usize, n_seq_max: usize) -> anyhow::Result<Ctx> {
        let raw = unsafe { kdn_new_ctx(model.raw, n_ctx as c_int, n_batch as c_int, n_seq_max as c_int) };
        anyhow::ensure!(!raw.is_null(), "创建 llama context 失败");
        Ok(Ctx { raw, n_vocab: model.n_vocab })
    }

    /// 在序列 seq 的位置 pos0 起喂入 toks；want_logits 时最后一个 token 出 logits。
    pub fn decode(&mut self, seq: i32, pos0: usize, toks: &[i32], want_logits: bool) -> anyhow::Result<()> {
        if toks.is_empty() {
            return Ok(());
        }
        let rc = unsafe { kdn_decode(self.raw, toks.as_ptr(), toks.len() as c_int, pos0 as c_int, seq, want_logits as c_int) };
        anyhow::ensure!(rc == 0, "llama_decode 失败 (rc={rc}, seq={seq}, pos0={pos0}, n={})", toks.len());
        Ok(())
    }

    /// 等 GPU 上排队的计算做完（计时用）
    pub fn sync(&mut self) {
        unsafe { kdn_sync(self.raw) }
    }

    /// 最近一次 decode 的最后一个 token 的 logits
    pub fn logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(kdn_logits_last(self.raw), self.n_vocab) }
    }

    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        unsafe { kdn_seq_rm(self.raw, seq, p0, p1) != 0 }
    }
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        unsafe { kdn_seq_cp(self.raw, src, dst, p0, p1) }
    }
    pub fn seq_pos_max(&self, seq: i32) -> i32 {
        unsafe { kdn_seq_pos_max(self.raw, seq) }
    }

    /// GPU -> 主机缓冲区（llama_state_seq_get_data）
    pub fn state_get(&mut self, seq: i32) -> Vec<u8> {
        let n = unsafe { kdn_state_seq_size(self.raw, seq) };
        let mut buf = vec![0u8; n];
        let w = unsafe { kdn_state_seq_get(self.raw, buf.as_mut_ptr(), n, seq) };
        buf.truncate(w);
        buf
    }

    /// 主机缓冲区 -> GPU；位置一起恢复，所以截掉前缀的部分状态也能恢复
    pub fn state_set(&mut self, seq: i32, data: &[u8]) -> bool {
        unsafe { kdn_state_seq_set(self.raw, data.as_ptr(), data.len(), seq) > 0 }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { kdn_free_ctx(self.raw) }
    }
}
