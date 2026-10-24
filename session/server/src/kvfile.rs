//! SSD 层：KV 文件读写。
//!
//! 格式：魔数 | 模型标识 | 覆盖的 token 序列 | llama_state_seq_get_data 的字节。
//! 带上 token 序列，恢复时和当前历史取公共前缀：回滚过、或崩溃前文件只覆盖一部分，都能用。
//! 读写都用 F_NOCACHE 绕过页缓存（真打到盘上），并按 1GB 分块（macOS 单次 read/write 不能超过 INT_MAX）。

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;

const MAGIC: &[u8; 8] = b"KDNKV001";
const CHUNK: usize = 1 << 30;

/// 主机内存里的一份 KV（DRAM 层），以及它覆盖的 token
pub struct HostKv {
    pub tokens: Vec<i32>,
    pub bytes: Vec<u8>,
    /// bytes[off..] 才是状态（从文件读回时前面是头部，省一次大块搬移）
    pub off: usize,
}

impl HostKv {
    pub fn state(&self) -> &[u8] {
        &self.bytes[self.off..]
    }
}

fn nocache(f: &File) {
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
}

/// 原子写：先写 tmp 再 rename；返回写入的字节数
pub fn write(path: &Path, model_id: &str, kv: &HostKv, tag: u64) -> std::io::Result<u64> {
    let tmp = path.with_extension(format!("tmp{tag}"));
    let mut hdr = Vec::with_capacity(32 + kv.tokens.len() * 4);
    hdr.extend_from_slice(MAGIC);
    hdr.extend_from_slice(&(model_id.len() as u32).to_le_bytes());
    hdr.extend_from_slice(model_id.as_bytes());
    hdr.extend_from_slice(&(kv.tokens.len() as u64).to_le_bytes());
    for t in &kv.tokens {
        hdr.extend_from_slice(&t.to_le_bytes());
    }
    hdr.extend_from_slice(&(kv.state().len() as u64).to_le_bytes());
    {
        let mut f = OpenOptions::new().write(true).create(true).truncate(true).open(&tmp)?;
        nocache(&f);
        f.write_all(&hdr)?;
        for c in kv.state().chunks(CHUNK) {
            f.write_all(c)?;
        }
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok((hdr.len() + kv.state().len()) as u64)
}

fn parse_header(buf: &[u8], model_id: &str) -> Result<(Vec<i32>, usize, u64), String> {
    let rd = |o: usize, n: usize| buf.get(o..o + n).ok_or_else(|| "KV 文件被截断".to_string());
    if rd(0, 8)? != MAGIC {
        return Err("魔数不对".into());
    }
    let ml = u32::from_le_bytes(rd(8, 4)?.try_into().unwrap()) as usize;
    let mid = rd(12, ml)?;
    if mid != model_id.as_bytes() {
        return Err(format!("KV 来自别的模型（{}）", String::from_utf8_lossy(mid)));
    }
    let mut o = 12 + ml;
    let n = u64::from_le_bytes(rd(o, 8)?.try_into().unwrap()) as usize;
    o += 8;
    let toks = rd(o, n * 4)?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    o += n * 4;
    let sl = u64::from_le_bytes(rd(o, 8)?.try_into().unwrap());
    Ok((toks, o + 8, sl))
}

/// 只读头部：覆盖的 token 数（启动时扫描用）
pub fn peek(path: &Path, model_id: &str) -> Result<usize, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 4096];
    let n = f.read(&mut buf).map_err(|e| e.to_string())?;
    buf.truncate(n);
    // 头部可能超过 4KB（token 多），够读出 token 数即可
    let ml = buf.get(8..12).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("太短")?;
    if buf.get(0..8) != Some(MAGIC) || buf.get(12..12 + ml) != Some(model_id.as_bytes()) {
        return Err("魔数或模型不匹配".into());
    }
    let n = buf.get(12 + ml..20 + ml).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("太短")?;
    Ok(n)
}

/// 整个读回（F_NOCACHE），返回 HostKv
pub fn read(path: &Path, model_id: &str) -> Result<HostKv, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    nocache(&f);
    let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
    let mut buf = vec![0u8; len];
    let mut off = 0;
    while off < len {
        let end = (off + CHUNK).min(len);
        let r = f.read(&mut buf[off..end]).map_err(|e| e.to_string())?;
        if r == 0 {
            break;
        }
        off += r;
    }
    let (tokens, o, sl) = parse_header(&buf[..off], model_id)?;
    if (o as u64 + sl) as usize != off {
        return Err(format!("KV 文件长度不对：{off} != {}", o as u64 + sl));
    }
    buf.truncate(off);
    Ok(HostKv { tokens, bytes: buf, off: o })
}
