//! 有状态会话引擎：单线程，独占 llama context，从命令队列取请求。
//!
//! 每个会话在服务端持有自己的 KV，位置在四种状态之一：
//!   gpu  ：llama.cpp KV cache 里的一个序列（slot），直接接着算
//!   dram ：llama_state_seq_get_data 拷到主机缓冲区
//!   ssd  ：KV 文件（F_NOCACHE）
//!   none ：只剩 token 历史，下次用时重算
//! token 历史永远保留（也落盘），所以 KV 丢了总能重算回来。
//!
//! 放置由客户端的等待提示驱动（见 policy.rs）：
//!   短工具等待 -> 留在 GPU；等人 -> 空闲一会儿后降到 DRAM；预测空档很长 -> 直接写 SSD；
//!   GPU / DRAM 满了，按"预测下次使用"最晚的先挪。

use crate::kvfile::{self, HostKv};
use crate::llama::{Ctx, Model};
use crate::policy::{evict_key, Predictor, Sig};
use crate::template;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc::UnboundedSender, oneshot};

pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e5).round() / 100.0
}

// ---------------------------------------------------------------- 配置

#[derive(Clone, Debug)]
pub struct Config {
    pub model_path: String,
    pub data_dir: PathBuf,
    pub n_ctx: usize,
    pub n_batch: usize,
    pub n_seq: usize,
    /// GPU 层容量（token），所有 GPU 会话加共享前缀
    pub gpu_tokens: usize,
    /// DRAM 层容量（token）
    pub dram_tokens: usize,
    /// 预测空档超过它直接降到 SSD（秒）
    pub cold_gap: f64,
    /// 等人时，空闲多久后从 GPU 降到 DRAM（秒）
    pub human_idle: f64,
    /// 在 DRAM 里空闲多久后写到 SSD（秒）
    pub dram_idle: f64,
    pub thinking: bool,
    pub prefix_cache: usize,
}

// ---------------------------------------------------------------- 会话

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Gpu,
    Dram,
    Ssd,
    None,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Gpu => "gpu",
            Tier::Dram => "dram",
            Tier::Ssd => "ssd",
            Tier::None => "none",
        }
    }
}

/// 每次 append 之前的位置，用于回滚
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnMark {
    pub tok: usize,
    pub msg: usize,
    pub pending_close: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wait {
    pub sig: Sig,
    pub tool: Option<String>,
    /// 等待开始（unix 秒）
    pub since: f64,
    /// 预测空档（秒）
    pub gap: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub max_tokens: Option<usize>,
}

/// 落盘的会话元数据（每轮写一次 JSON）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    pub label: String,
    pub model_id: String,
    pub created: f64,
    pub updated: f64,
    pub thinking: bool,
    pub system_len: usize,
    /// 已经进了 KV 的全部 token（从位置 0 起）
    pub tokens: Vec<i32>,
    /// 上一轮生成停在 <|im_end|> 前，下一轮增量先补上
    pub pending_close: bool,
    /// 对话记录（OpenAI 风格），只用于展示 / 回滚 / 统计，不参与推理
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub turns: Vec<TurnMark>,
    pub pred: Predictor,
    pub wait: Option<Wait>,
    pub sampling: Sampling,
}

pub struct Session {
    pub m: Meta,
    tier: Tier,
    slot: i32,
    dram: Option<Arc<HostKv>>,
    /// 后台写 SSD 的代号；完成消息带回来对得上才算数
    flush_gen: u64,
    flushing: bool,
    /// SSD 上有一份 KV 文件，覆盖多少 token（可能只是历史的前缀）
    ssd_tokens: Option<usize>,
}

impl Session {
    fn new(m: Meta) -> Session {
        Session { m, tier: Tier::None, slot: -1, dram: None, flush_gen: 0, flushing: false, ssd_tokens: None }
    }
    fn footprint(&self) -> usize {
        match self.tier {
            Tier::Gpu => self.m.tokens.len(),
            Tier::Dram => self.dram.as_ref().map_or(0, |d| d.tokens.len()),
            _ => 0,
        }
    }
    /// 驱逐键：越大越先挪走
    fn key(&self, now: f64) -> f64 {
        match &self.m.wait {
            Some(w) => evict_key(now, w.since, w.gap),
            None => evict_key(now, self.m.updated, 60.0),
        }
    }
}

// ---------------------------------------------------------------- 命令

pub type Reply = oneshot::Sender<Result<Value, ApiErr>>;

#[derive(Debug)]
pub struct ApiErr(pub u16, pub String);

fn bad(msg: impl Into<String>) -> ApiErr {
    ApiErr(400, msg.into())
}
fn not_found(id: &str) -> ApiErr {
    ApiErr(404, format!("没有会话 {id}"))
}

pub enum Event {
    Token(String),
    Done(Value),
    Error(String),
}

#[derive(Deserialize, Default)]
pub struct CreateReq {
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub thinking: Option<bool>,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub sampling: Sampling,
}

#[derive(Deserialize, Default)]
pub struct HintReq {
    pub waiting: String,
    pub tool: Option<String>,
    pub expected_s: Option<f64>,
}

#[derive(Deserialize, Default)]
pub struct AppendReq {
    pub messages: Vec<Value>,
    #[serde(flatten)]
    pub sampling: Sampling,
    /// 生成结束后要应用的提示；不给就由服务端推断（有工具调用 -> tool，否则 -> human）
    pub hint: Option<HintReq>,
    /// 返回生成的 token id（正确性对比用）
    #[serde(default)]
    pub debug: bool,
    #[serde(skip)]
    pub request_bytes: usize,
}

pub enum Cmd {
    Create(CreateReq, Reply),
    Append(String, AppendReq, UnboundedSender<Event>),
    Hint(String, HintReq, Reply),
    Fork(String, Option<usize>, Reply),
    Rollback(String, usize, Reply),
    Get(String, bool, Reply),
    List(Reply),
    Delete(String, Reply),
    Demote(String, String, Reply),
    Stats(Reply),
    Flushed { id: String, gen: u64, result: Result<(usize, u64), String> },
    Shutdown(Reply),
}

// ---------------------------------------------------------------- 引擎

struct Prefix {
    tokens: Vec<i32>,
    slot: i32,
    last_used: f64,
}

/// 一次请求开始时把 KV 弄回 GPU 的统计
#[derive(Default)]
struct Restore {
    where_: &'static str,
    restore_ms: f64,
    recompute_tokens: usize,
    recompute_ms: f64,
    prefix_hit_tokens: usize,
}

pub struct Engine {
    pub model: Model,
    pub ctx: Ctx,
    pub cfg: Config,
    pub model_id: String,
    sessions: HashMap<String, Session>,
    free_slots: Vec<i32>,
    prefixes: Vec<Prefix>,
    tx_self: Sender<Cmd>,
    next_gen: u64,
    /// 后台 I/O 累计（秒），不在请求关键路径上
    bg_io_s: f64,
}

impl Engine {
    pub fn new(cfg: Config, tx_self: Sender<Cmd>) -> anyhow::Result<Engine> {
        let model = Model::load(&cfg.model_path, 999)?;
        let ctx = Ctx::new(&model, cfg.n_ctx, cfg.n_batch, cfg.n_seq)?;
        let size = std::fs::metadata(&cfg.model_path)?.len();
        let name = std::path::Path::new(&cfg.model_path).file_name().unwrap().to_string_lossy().into_owned();
        let model_id = format!("{name}:{size}");
        let free_slots = (0..cfg.n_seq as i32).rev().collect();
        std::fs::create_dir_all(&cfg.data_dir)?;
        let mut e = Engine {
            model,
            ctx,
            cfg,
            model_id,
            sessions: HashMap::new(),
            free_slots,
            prefixes: vec![],
            tx_self,
            next_gen: 1,
            bg_io_s: 0.0,
        };
        e.load_all();
        Ok(e)
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.cfg.data_dir.join(format!("{id}.json"))
    }
    fn kv_path(&self, id: &str) -> PathBuf {
        self.cfg.data_dir.join(format!("{id}.kv"))
    }

    /// 启动时把磁盘上的会话读回来：元数据 + token 历史；KV 文件头校验通过的算在 SSD 层
    fn load_all(&mut self) {
        let Ok(rd) = std::fs::read_dir(&self.cfg.data_dir) else { return };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let m: Meta = match std::fs::read(&p).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[load] 跳过 {}: {e}", p.display());
                    continue;
                }
            };
            let mut s = Session::new(m);
            let kvp = self.kv_path(&s.m.id);
            if kvp.exists() {
                match kvfile::peek(&kvp, &self.model_id) {
                    Ok(n) => {
                        s.ssd_tokens = Some(n);
                        s.tier = Tier::Ssd;
                    }
                    Err(e) => eprintln!("[load] {} 的 KV 文件不可用（{e}），下次用时重算", s.m.id),
                }
            }
            if s.m.model_id != self.model_id {
                eprintln!("[load] {} 由 {} 创建，当前模型 {}：KV 将从 token 历史重算", s.m.id, s.m.model_id, self.model_id);
            }
            self.sessions.insert(s.m.id.clone(), s);
        }
        eprintln!("[load] 从 {} 读回 {} 个会话", self.cfg.data_dir.display(), self.sessions.len());
    }

    fn save_meta(&self, s: &Session) {
        let p = self.meta_path(&s.m.id);
        let tmp = p.with_extension("json.tmp");
        let r = serde_json::to_vec(&s.m).map_err(std::io::Error::other).and_then(|b| std::fs::write(&tmp, b)).and_then(|_| std::fs::rename(&tmp, &p));
        if let Err(e) = r {
            eprintln!("[save] {}: {e}", s.m.id);
        }
    }

    // ------------------------------------------------------------ 主循环

    pub fn run(mut self, rx: Receiver<Cmd>) {
        loop {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(Cmd::Shutdown(r)) => {
                    let v = self.shutdown();
                    let _ = r.send(Ok(v));
                    return;
                }
                Ok(cmd) => self.handle(cmd),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            self.tick(unix_now());
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Create(req, r) => {
                let _ = r.send(self.create(req));
            }
            Cmd::Append(id, req, tx) => self.append(&id, req, tx),
            Cmd::Hint(id, h, r) => {
                let _ = r.send(self.hint(&id, h));
            }
            Cmd::Fork(id, turn, r) => {
                let _ = r.send(self.fork(&id, turn));
            }
            Cmd::Rollback(id, n, r) => {
                let _ = r.send(self.rollback(&id, n));
            }
            Cmd::Get(id, msgs, r) => {
                let _ = r.send(self.sessions.get(&id).map(|s| self.summary(s, msgs)).ok_or_else(|| not_found(&id)));
            }
            Cmd::List(r) => {
                let mut v: Vec<&Session> = self.sessions.values().collect();
                v.sort_by(|a, b| b.m.updated.total_cmp(&a.m.updated));
                let _ = r.send(Ok(json!({"sessions": v.iter().map(|s| self.summary(s, false)).collect::<Vec<_>>()})));
            }
            Cmd::Delete(id, r) => {
                let _ = r.send(self.delete(&id));
            }
            Cmd::Demote(id, tier, r) => {
                let _ = r.send(self.admin_demote(&id, &tier));
            }
            Cmd::Stats(r) => {
                let _ = r.send(Ok(self.stats()));
            }
            Cmd::Flushed { id, gen, result } => self.on_flushed(&id, gen, result),
            Cmd::Shutdown(_) => unreachable!(),
        }
    }

    pub fn summary(&self, s: &Session, with_messages: bool) -> Value {
        let mut v = json!({
            "id": s.m.id,
            "label": s.m.label,
            "tokens": s.m.tokens.len(),
            "turns": s.m.turns.len(),
            "tier": if s.flushing { "dram(flushing->ssd)" } else { s.tier.name() },
            "ssd_tokens": s.ssd_tokens,
            "thinking": s.m.thinking,
            "created": s.m.created,
            "updated": s.m.updated,
            "waiting": s.m.wait,
            "model_id": s.m.model_id,
            "n_messages": s.m.messages.len(),
        });
        if with_messages {
            v["messages"] = Value::Array(s.m.messages.clone());
            v["tools"] = Value::Array(s.m.tools.clone());
        }
        v
    }

    fn stats(&self) -> Value {
        let mut used = [0usize; 3];
        let mut count = [0usize; 4];
        for s in self.sessions.values() {
            match s.tier {
                Tier::Gpu => used[0] += s.footprint(),
                Tier::Dram => used[1] += s.footprint(),
                _ => {}
            }
            if let Some(n) = s.ssd_tokens {
                used[2] += n;
            }
            count[s.tier as usize] += 1;
        }
        json!({
            "model": self.model_id,
            "n_ctx": self.cfg.n_ctx,
            "sessions": self.sessions.len(),
            "count": {"gpu": count[0], "dram": count[1], "ssd": count[2], "none": count[3]},
            "used_tokens": {"gpu": used[0] + self.prefix_tokens(), "dram": used[1], "ssd_files": used[2]},
            "cap_tokens": {"gpu": self.cfg.gpu_tokens, "dram": self.cfg.dram_tokens},
            "prefix_cache": self.prefixes.iter().map(|p| p.tokens.len()).collect::<Vec<_>>(),
            "free_slots": self.free_slots.len(),
            "background_io_s": self.bg_io_s,
        })
    }

    // ------------------------------------------------------------ 层间搬运

    fn prefix_tokens(&self) -> usize {
        self.prefixes.iter().map(|p| p.tokens.len()).sum()
    }

    fn gpu_used(&self) -> usize {
        self.sessions.values().filter(|s| s.tier == Tier::Gpu).map(|s| s.m.tokens.len()).sum::<usize>() + self.prefix_tokens()
    }

    fn dram_used(&self) -> usize {
        self.sessions.values().filter(|s| s.tier == Tier::Dram).map(|s| s.footprint()).sum()
    }

    /// GPU 腾出 need 个 token（need_slot 时还要一个空闲 slot）。
    /// incoming_key：调用方自己的驱逐键；只挪比它更晚才用的（放置决策用）。
    /// 正在处理的会话已经从表里取出来了，不会被挑中。
    fn make_room_gpu(&mut self, need: usize, need_slot: bool, incoming_key: Option<f64>) -> bool {
        let now = unix_now();
        loop {
            let fits = self.gpu_used() + need <= self.cfg.gpu_tokens;
            if fits && (!need_slot || !self.free_slots.is_empty()) {
                return true;
            }
            let victim = self
                .sessions
                .values()
                .filter(|s| s.tier == Tier::Gpu)
                .map(|s| (s.key(now), s.m.id.clone()))
                .max_by(|a, b| a.0.total_cmp(&b.0));
            match victim {
                Some((k, id)) => {
                    if incoming_key.is_some_and(|ik| k <= ik) {
                        return false; // 留下的都比我更早要用
                    }
                    let mut v = self.sessions.remove(&id).unwrap();
                    self.to_dram(&mut v);
                    self.sessions.insert(id, v);
                    self.make_room_dram();
                }
                None => {
                    // 没有会话可挪了：丢掉最久没用的共享前缀
                    if let Some(i) = (0..self.prefixes.len()).min_by(|&a, &b| self.prefixes[a].last_used.total_cmp(&self.prefixes[b].last_used)) {
                        let p = self.prefixes.remove(i);
                        self.ctx.seq_rm(p.slot, -1, -1);
                        self.free_slots.push(p.slot);
                        continue;
                    }
                    // 会话本身比 GPU 层还大：只要 slot 有就硬上（n_ctx 才是硬上限）
                    return incoming_key.is_none() && !self.free_slots.is_empty();
                }
            }
        }
    }

    /// DRAM 超容量：驱逐键最大的几个后台写 SSD
    fn make_room_dram(&mut self) {
        let now = unix_now();
        let mut over = self.dram_used().saturating_sub(self.cfg.dram_tokens) as isize;
        if over <= 0 {
            return;
        }
        let mut cands: Vec<(f64, String, usize)> = self
            .sessions
            .values()
            .filter(|s| s.tier == Tier::Dram && !s.flushing)
            .map(|s| (s.key(now), s.m.id.clone(), s.footprint()))
            .collect();
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, id, size) in cands {
            if over <= 0 {
                break;
            }
            let mut s = self.sessions.remove(&id).unwrap();
            self.start_flush(&mut s);
            self.sessions.insert(id, s);
            over -= size as isize;
        }
    }

    fn alloc_slot(&mut self) -> Option<i32> {
        self.free_slots.pop()
    }

    /// GPU -> DRAM：llama_state_seq_get_data 拷到主机，释放 slot
    fn to_dram(&mut self, s: &mut Session) {
        if s.tier != Tier::Gpu {
            return;
        }
        let bytes = self.ctx.state_get(s.slot);
        self.ctx.seq_rm(s.slot, -1, -1);
        self.free_slots.push(s.slot);
        s.slot = -1;
        s.dram = Some(Arc::new(HostKv { tokens: s.m.tokens.clone(), bytes, off: 0 }));
        s.tier = Tier::Dram;
    }

    /// DRAM -> SSD，后台线程写；写完之前会话仍在 DRAM，可以随时被恢复
    fn start_flush(&mut self, s: &mut Session) {
        if s.tier == Tier::Gpu {
            self.to_dram(s);
        }
        let Some(kv) = s.dram.clone() else { return };
        let gen = self.next_gen;
        self.next_gen += 1;
        s.flush_gen = gen;
        s.flushing = true;
        let (path, model_id, id, tx) = (self.kv_path(&s.m.id), self.model_id.clone(), s.m.id.clone(), self.tx_self.clone());
        std::thread::spawn(move || {
            let result = kvfile::write(&path, &model_id, &kv, gen).map(|b| (kv.tokens.len(), b)).map_err(|e| e.to_string());
            let _ = tx.send(Cmd::Flushed { id, gen, result });
        });
    }

    fn on_flushed(&mut self, id: &str, gen: u64, result: Result<(usize, u64), String>) {
        let Some(s) = self.sessions.get_mut(id) else {
            // 写的过程中会话被删了：别留孤儿文件
            let _ = std::fs::remove_file(self.kv_path(id));
            return;
        };
        if s.flush_gen != gen {
            return;
        }
        s.flushing = false;
        match result {
            Ok((n, _bytes)) => {
                s.ssd_tokens = Some(n);
                if s.tier == Tier::Dram {
                    s.dram = None;
                    s.tier = Tier::Ssd;
                }
            }
            Err(e) => eprintln!("[flush] {id}: {e}（留在 DRAM）"),
        }
    }

    /// 同步写 SSD（管理接口和关机用）
    fn to_ssd_sync(&mut self, s: &mut Session) -> Result<u64, String> {
        self.to_dram(s);
        let Some(kv) = s.dram.clone() else {
            return Ok(0); // 本来就不在 GPU/DRAM
        };
        let gen = self.next_gen;
        self.next_gen += 1;
        s.flush_gen = gen; // 让还在飞的后台写作废
        s.flushing = false;
        let b = kvfile::write(&self.kv_path(&s.m.id), &self.model_id, &kv, gen).map_err(|e| e.to_string())?;
        s.ssd_tokens = Some(kv.tokens.len());
        s.dram = None;
        s.tier = Tier::Ssd;
        Ok(b)
    }

    fn drop_kv(&mut self, s: &mut Session) {
        if s.tier == Tier::Gpu {
            self.ctx.seq_rm(s.slot, -1, -1);
            self.free_slots.push(s.slot);
            s.slot = -1;
        }
        s.dram = None;
        s.flush_gen = 0;
        s.flushing = false;
        let _ = std::fs::remove_file(self.kv_path(&s.m.id));
        s.ssd_tokens = None;
        s.tier = Tier::None;
    }

    /// 在 slot 上从位置 pos0 起 prefill
    fn prefill(&mut self, slot: i32, pos0: usize, toks: &[i32], logits: bool) -> Result<(), ApiErr> {
        self.ctx.decode(slot, pos0, toks, logits).map_err(|e| ApiErr(500, e.to_string()))?;
        self.ctx.sync(); // 异步 decode：同步后计时才准
        Ok(())
    }

    /// 把会话的 KV 弄回 GPU（extra：接下来还要追加的 token 数）
    fn ensure_gpu(&mut self, s: &mut Session, extra: usize) -> Result<Restore, ApiErr> {
        let mut r = Restore { where_: s.tier.name(), ..Default::default() };
        if s.tier == Tier::Gpu {
            return Ok(r);
        }
        let n = s.m.tokens.len();
        if !self.make_room_gpu(n + extra, true, None) {
            return Err(ApiErr(503, "GPU 上没有空闲序列".into()));
        }
        let slot = self.alloc_slot().ok_or_else(|| ApiErr(503, "没有空闲序列".into()))?;
        let t0 = Instant::now();
        // DRAM / SSD：拿到主机缓冲区
        let host: Option<Arc<HostKv>> = match s.tier {
            Tier::Dram => s.dram.clone(),
            Tier::Ssd => match kvfile::read(&self.kv_path(&s.m.id), &self.model_id) {
                Ok(kv) => Some(Arc::new(kv)),
                Err(e) => {
                    eprintln!("[restore] {} 读 SSD 失败（{e}），改为重算", s.m.id);
                    r.where_ = "ssd-failed";
                    None
                }
            },
            _ => None,
        };
        let mut have = 0; // slot 里已有的前缀长度
        if let Some(kv) = host {
            if self.ctx.state_set(slot, kv.state()) {
                // 和当前历史取公共前缀：回滚过的话 KV 比历史长，截掉多余部分
                let c = kv.tokens.iter().zip(&s.m.tokens).take_while(|(a, b)| a == b).count();
                if c < kv.tokens.len() {
                    self.ctx.seq_rm(slot, c as i32, -1);
                }
                have = c;
                debug_assert_eq!(self.ctx.seq_pos_max(slot), c as i32 - 1);
            } else {
                eprintln!("[restore] {} llama_state_seq_set_data 失败，改为重算", s.m.id);
                self.ctx.seq_rm(slot, -1, -1);
            }
            r.restore_ms = ms(t0.elapsed());
        }
        if have == 0 && self.cfg.prefix_cache > 0 {
            // 冷启动 / 丢失：先挂上共享前缀（系统提示 + 工具定义）
            let now = unix_now();
            if let Some(p) = self.prefixes.iter_mut().filter(|p| p.tokens.len() <= n && s.m.tokens.starts_with(&p.tokens)).max_by_key(|p| p.tokens.len()) {
                self.ctx.seq_cp(p.slot, slot, -1, -1);
                p.last_used = now;
                have = p.tokens.len();
                r.prefix_hit_tokens = have;
            }
        }
        if have < n {
            let t1 = Instant::now();
            let toks = s.m.tokens[have..].to_vec();
            if let Err(e) = self.prefill(slot, have, &toks, false) {
                self.ctx.seq_rm(slot, -1, -1);
                self.free_slots.push(slot);
                return Err(e);
            }
            r.recompute_tokens = n - have;
            r.recompute_ms = ms(t1.elapsed());
        }
        s.slot = slot;
        s.tier = Tier::Gpu;
        s.dram = None;
        s.flushing = false;
        s.flush_gen = 0; // 在飞的后台写回来后只记文件，不再改层
        if r.where_ == "none" {
            r.where_ = "recompute";
        }
        Ok(r)
    }

    /// 新会话的系统前缀登记为共享前缀（seq_cp 只是共享 KV 单元，不复制）
    fn register_prefix(&mut self, slot: i32, toks: &[i32]) {
        if self.cfg.prefix_cache == 0 || self.prefixes.iter().any(|p| p.tokens == toks) {
            return;
        }
        if self.prefixes.len() >= self.cfg.prefix_cache {
            let i = (0..self.prefixes.len()).min_by(|&a, &b| self.prefixes[a].last_used.total_cmp(&self.prefixes[b].last_used)).unwrap();
            let p = self.prefixes.remove(i);
            self.ctx.seq_rm(p.slot, -1, -1);
            self.free_slots.push(p.slot);
        }
        let Some(ps) = self.alloc_slot() else { return };
        self.ctx.seq_cp(slot, ps, 0, toks.len() as i32);
        self.prefixes.push(Prefix { tokens: toks.to_vec(), slot: ps, last_used: unix_now() });
    }

    // ------------------------------------------------------------ 放置

    /// 按等待提示决定放在哪一层
    fn place(&mut self, s: &mut Session) -> &'static str {
        let Some(w) = s.m.wait.clone() else { return "none" };
        if w.gap > self.cfg.cold_gap {
            if s.tier == Tier::Gpu || s.tier == Tier::Dram {
                self.start_flush(s);
                return "ssd";
            }
            return "keep";
        }
        if s.tier != Tier::Gpu {
            return "keep";
        }
        let now = unix_now();
        if self.make_room_gpu(s.m.tokens.len(), false, Some(s.key(now))) {
            if w.sig.is_human() {
                "gpu(human idle timer)"
            } else {
                "gpu"
            }
        } else {
            self.to_dram(s);
            self.make_room_dram();
            "dram"
        }
    }

    /// 定时：等人的会话空闲一会儿降到 DRAM；DRAM 里空闲久了写 SSD；超时很久的工具等待也降级
    fn tick(&mut self, now: f64) {
        let ids: Vec<String> = self.sessions.keys().cloned().collect();
        for id in ids {
            let mut s = self.sessions.remove(&id).unwrap();
            let (since, gap, human) = match &s.m.wait {
                Some(w) => (w.since, w.gap, w.sig.is_human()),
                None => (s.m.updated, 60.0, true),
            };
            let idle = now - since;
            match s.tier {
                Tier::Gpu if (human && idle > self.cfg.human_idle) || (!human && idle > (2.0 * gap).max(gap + 30.0)) => {
                    let t = Instant::now();
                    self.to_dram(&mut s);
                    self.bg_io_s += t.elapsed().as_secs_f64();
                    eprintln!("[tier] {id}: gpu -> dram（空闲 {idle:.0}s）");
                    self.sessions.insert(id, s);
                    self.make_room_dram();
                    continue;
                }
                Tier::Dram if !s.flushing && idle > self.cfg.dram_idle => {
                    eprintln!("[tier] {id}: dram -> ssd（空闲 {idle:.0}s）");
                    self.start_flush(&mut s);
                }
                _ => {}
            }
            self.sessions.insert(id, s);
        }
    }

    // ------------------------------------------------------------ API 操作

    fn new_id(&self) -> String {
        loop {
            let mut b = [0u8; 4];
            if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
                use std::io::Read;
                let _ = f.read_exact(&mut b);
            }
            let id = format!("s{:08x}", u32::from_le_bytes(b));
            if !self.sessions.contains_key(&id) {
                return id;
            }
        }
    }

    fn create(&mut self, req: CreateReq) -> Result<Value, ApiErr> {
        let text = template::system_text(&req.system, &req.tools);
        let toks = self.model.tokenize(&text, self.model.add_bos());
        if toks.len() + 256 > self.cfg.n_ctx {
            return Err(bad(format!("系统提示 {} token，超过上下文", toks.len())));
        }
        let now = unix_now();
        let m = Meta {
            id: self.new_id(),
            label: req.label,
            model_id: self.model_id.clone(),
            created: now,
            updated: now,
            thinking: req.thinking.unwrap_or(self.cfg.thinking),
            system_len: toks.len(),
            tokens: toks.clone(),
            pending_close: false,
            messages: vec![json!({"role": "system", "content": req.system})],
            tools: req.tools,
            turns: vec![],
            pred: Predictor::default(),
            wait: None,
            sampling: req.sampling,
        };
        let mut s = Session::new(m);
        // 和 recompute 走同一条路：命中共享前缀就 seq_cp，否则 prefill
        let r = self.ensure_gpu(&mut s, 0)?;
        self.register_prefix(s.slot, &toks);
        self.save_meta(&s);
        let mut v = self.summary(&s, false);
        v["stats"] = json!({"system_tokens": toks.len(), "prefill_ms": r.recompute_ms, "prefix_hit_tokens": r.prefix_hit_tokens});
        self.sessions.insert(s.m.id.clone(), s);
        Ok(v)
    }

    fn append(&mut self, id: &str, req: AppendReq, tx: UnboundedSender<Event>) {
        let Some(mut s) = self.sessions.remove(id) else {
            let _ = tx.send(Event::Error(format!("没有会话 {id}")));
            return;
        };
        let r = self.append_inner(&mut s, req, &tx);
        if let Err(e) = &r {
            let _ = tx.send(Event::Error(e.1.clone()));
        }
        self.save_meta(&s);
        self.sessions.insert(id.to_string(), s);
    }

    fn append_inner(&mut self, s: &mut Session, req: AppendReq, tx: &UnboundedSender<Event>) -> Result<(), ApiErr> {
        let t_req = Instant::now();
        let now = unix_now();
        // 1. 构造增量（先校验，失败不动会话）
        let delta_text = template::delta_text(s.m.pending_close, &req.messages, s.m.thinking).map_err(bad)?;
        let delta = self.model.tokenize(&delta_text, false);
        let sp = merge_sampling(&req.sampling, &s.m.sampling, s.m.thinking);
        let room = self.cfg.n_ctx.saturating_sub(s.m.tokens.len() + delta.len() + 1);
        if room < 16 {
            return Err(bad(format!("上下文已满（{} token，n_ctx={}），请 rollback 或新开会话", s.m.tokens.len(), self.cfg.n_ctx)));
        }
        let max_tokens = sp.max_tokens.unwrap().min(room);

        // 2. 上一次等待的真实空档，喂给预测器
        let waited = s.m.wait.take().map(|w| {
            let gap = now - w.since;
            s.m.pred.observe(w.sig, gap);
            json!({"sig": w.sig, "predicted_s": w.gap, "actual_s": (gap * 10.0).round() / 10.0})
        });

        // 3. KV 回到 GPU
        let rs = self.ensure_gpu(s, delta.len() + max_tokens)?;

        // 4. 只 prefill 增量
        s.m.turns.push(TurnMark { tok: s.m.tokens.len(), msg: s.m.messages.len(), pending_close: s.m.pending_close });
        let tp = Instant::now();
        let pos0 = s.m.tokens.len();
        if let Err(e) = self.prefill(s.slot, pos0, &delta, true) {
            self.ctx.seq_rm(s.slot, pos0 as i32, -1);
            s.m.turns.pop();
            return Err(e);
        }
        let prefill_ms = ms(tp.elapsed());
        s.m.tokens.extend_from_slice(&delta);
        s.m.messages.extend(req.messages.iter().cloned());

        // 5. 生成：采样出来的 token 直接进 KV，不重新分词
        let mut rng = Rng::new(sp.seed.unwrap_or_else(|| (now * 1e6) as u64));
        let mut gen: Vec<i32> = vec![];
        let mut utf8 = Utf8Stream::default();
        let mut ttft_ms = 0.0;
        let td = Instant::now();
        let mut stop = "length";
        for i in 0..max_tokens {
            let t = sample(self.ctx.logits(), &sp, &gen, &mut rng);
            if i == 0 {
                ttft_ms = ms(t_req.elapsed());
            }
            if self.model.is_eog(t) {
                stop = "eos";
                break;
            }
            gen.push(t);
            if let Some(txt) = utf8.push(&self.model.piece(t)) {
                let _ = tx.send(Event::Token(txt));
            }
            let pos = s.m.tokens.len();
            if let Err(e) = self.prefill(s.slot, pos, &[t], true) {
                eprintln!("[gen] {}: {}", s.m.id, e.1);
                stop = "error";
                break;
            }
            s.m.tokens.push(t);
        }
        if let Some(txt) = utf8.flush() {
            let _ = tx.send(Event::Token(txt));
        }
        let decode_s = td.elapsed().as_secs_f64();
        s.m.pending_close = true;

        // 6. 解析：思考内容、正文、工具调用
        let text = self.model.detokenize(&gen);
        let (reasoning, content, tool_calls) = parse_output(&text, s.m.turns.len());
        let mut amsg = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            amsg["reasoning_content"] = json!(reasoning);
        }
        if !tool_calls.is_empty() {
            amsg["tool_calls"] = json!(tool_calls);
        }
        s.m.messages.push(amsg);
        s.m.updated = unix_now();

        // 7. 等待提示：请求里给了就用，否则按输出推断
        let hint = req.hint.unwrap_or_else(|| {
            if let Some(tc) = tool_calls.first() {
                HintReq { waiting: "tool".into(), tool: tc["function"]["name"].as_str().map(String::from), expected_s: None }
            } else {
                HintReq { waiting: "human".into(), tool: None, expected_s: None }
            }
        });
        let placed = self.apply_hint(s, &hint);

        let gen_n = gen.len();
        let mut stats = json!({
            "kv_tier": rs.where_,
            "restore_ms": rs.restore_ms,
            "recompute_tokens": rs.recompute_tokens,
            "recompute_ms": rs.recompute_ms,
            "prefix_hit_tokens": rs.prefix_hit_tokens,
            "new_tokens": delta.len(),
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "gen_tokens": gen_n,
            "decode_ms": ms(Duration::from_secs_f64(decode_s)),
            "decode_tok_s": if decode_s > 0.0 { ((gen_n as f64 / decode_s) * 10.0).round() / 10.0 } else { 0.0 },
            "total_tokens": s.m.tokens.len(),
            "request_bytes": req.request_bytes,
            "stop": stop,
        });
        if let Some(w) = waited {
            stats["waited"] = w;
        }
        let mut done = json!({
            "id": s.m.id,
            "turn": s.m.turns.len(),
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "stats": stats,
            "hint": placed,
        });
        if req.debug {
            done["token_ids"] = json!(gen);
        }
        let _ = tx.send(Event::Done(done));
        Ok(())
    }

    fn apply_hint(&mut self, s: &mut Session, h: &HintReq) -> Value {
        let Some(sig) = Sig::from_hint(&h.waiting, h.tool.as_deref()) else {
            return json!({"error": format!("未知 waiting: {}", h.waiting)});
        };
        let gap = h.expected_s.unwrap_or_else(|| s.m.pred.gap(sig));
        s.m.wait = Some(Wait { sig, tool: h.tool.clone(), since: unix_now(), gap });
        let action = self.place(s);
        json!({"waiting": h.waiting, "tool": h.tool, "sig": sig, "predicted_gap_s": gap, "placement": action, "tier": s.tier.name()})
    }

    fn hint(&mut self, id: &str, h: HintReq) -> Result<Value, ApiErr> {
        let mut s = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        if Sig::from_hint(&h.waiting, h.tool.as_deref()).is_none() {
            self.sessions.insert(id.into(), s);
            return Err(bad(format!("waiting 只能是 tool / human / ask / done，收到 {}", h.waiting)));
        }
        let sig = Sig::from_hint(&h.waiting, h.tool.as_deref()).unwrap();
        let prev = s.m.wait.clone();
        let v = match prev {
            // 和服务端推断的一样（append 结束时已按输出推断过）：不重置等待起点，只更新预计时长
            Some(w) if w.sig == sig && w.tool == h.tool => {
                if let Some(e) = h.expected_s {
                    s.m.wait.as_mut().unwrap().gap = e;
                }
                let placed = self.place(&mut s);
                json!({"waiting": h.waiting, "tool": h.tool, "sig": sig, "predicted_gap_s": s.m.wait.as_ref().unwrap().gap, "placement": placed, "tier": s.tier.name()})
            }
            _ => {
                // 换了等待对象（例如人确认完、开始跑 bash）：前一段等人的实际时长喂给预测器
                if let Some(w) = prev.filter(|w| w.sig.is_human()) {
                    s.m.pred.observe(w.sig, unix_now() - w.since);
                }
                self.apply_hint(&mut s, &h)
            }
        };
        self.save_meta(&s);
        self.sessions.insert(id.into(), s);
        Ok(v)
    }

    fn fork(&mut self, id: &str, turn: Option<usize>) -> Result<Value, ApiErr> {
        let mut p = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        let t0 = Instant::now();
        let r = self.ensure_gpu(&mut p, 0);
        let rs = match r {
            Ok(r) => r,
            Err(e) => {
                self.sessions.insert(id.into(), p);
                return Err(e);
            }
        };
        let restore_ms = ms(t0.elapsed());
        let mut m = p.m.clone();
        m.id = self.new_id();
        m.label = format!("fork of {id}");
        m.created = unix_now();
        m.updated = m.created;
        m.wait = None;
        let mut c = Session::new(m);
        // 子会话：先保证有 slot，再 seq_cp（kv_unified 下共享 KV 单元，不复制）。
        // 父会话此时不在表里，不会被挤掉。
        let need = c.m.tokens.len();
        if self.make_room_gpu(need, true, None) {
            if let Some(slot) = self.alloc_slot() {
                self.ctx.seq_cp(p.slot, slot, -1, -1);
                c.slot = slot;
                c.tier = Tier::Gpu;
            }
        }
        // 拿不到 slot：子会话只带 token 历史，用时重算
        self.sessions.insert(id.into(), p);
        let fork_ms = ms(t0.elapsed());
        if let Some(n) = turn {
            self.truncate(&mut c, n)?;
        }
        self.save_meta(&c);
        let mut v = self.summary(&c, false);
        v["stats"] = json!({"parent_kv_tier": rs.where_, "parent_restore_ms": restore_ms, "fork_ms": fork_ms, "shared_tokens": need});
        self.sessions.insert(c.m.id.clone(), c);
        Ok(v)
    }

    fn truncate(&mut self, s: &mut Session, n: usize) -> Result<(), ApiErr> {
        if n > s.m.turns.len() {
            return Err(bad(format!("只有 {} 轮", s.m.turns.len())));
        }
        if n == s.m.turns.len() {
            return Ok(());
        }
        let mark = s.m.turns[n].clone();
        s.m.tokens.truncate(mark.tok);
        s.m.messages.truncate(mark.msg);
        s.m.pending_close = mark.pending_close;
        s.m.turns.truncate(n);
        if s.tier == Tier::Gpu {
            self.ctx.seq_rm(s.slot, mark.tok as i32, -1);
        }
        // DRAM / SSD 里的 KV 比历史长：恢复时按公共前缀截掉
        s.m.updated = unix_now();
        Ok(())
    }

    fn rollback(&mut self, id: &str, n: usize) -> Result<Value, ApiErr> {
        let mut s = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        let before = s.m.tokens.len();
        let r = self.truncate(&mut s, n);
        self.save_meta(&s);
        let mut v = self.summary(&s, false);
        v["removed_tokens"] = json!(before - s.m.tokens.len());
        self.sessions.insert(id.into(), s);
        r.map(|_| v)
    }

    fn delete(&mut self, id: &str) -> Result<Value, ApiErr> {
        let mut s = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        self.drop_kv(&mut s);
        let _ = std::fs::remove_file(self.meta_path(id));
        Ok(json!({"deleted": id}))
    }

    /// 管理接口：强制降级，测试用。tier = dram | ssd | drop（丢掉 KV，只留 token 历史）| gpu
    fn admin_demote(&mut self, id: &str, tier: &str) -> Result<Value, ApiErr> {
        let mut s = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        let from = s.tier.name();
        let t0 = Instant::now();
        let mut bytes = 0u64;
        let r = match tier {
            "dram" => {
                if s.tier == Tier::Ssd || s.tier == Tier::None {
                    Err(bad("只能从 gpu 降到 dram"))
                } else {
                    self.to_dram(&mut s);
                    bytes = s.dram.as_ref().map_or(0, |d| d.bytes.len() as u64);
                    Ok(())
                }
            }
            "ssd" => self.to_ssd_sync(&mut s).map(|b| bytes = b).map_err(|e| ApiErr(500, e)),
            "drop" => {
                self.drop_kv(&mut s);
                Ok(())
            }
            "gpu" => self.ensure_gpu(&mut s, 0).map(|_| ()),
            _ => Err(bad("tier 只能是 gpu / dram / ssd / drop")),
        };
        let v = json!({"id": id, "from": from, "to": s.tier.name(), "ms": ms(t0.elapsed()), "bytes": bytes, "tokens": s.m.tokens.len()});
        self.sessions.insert(id.into(), s);
        r.map(|_| v)
    }

    /// 关机：GPU / DRAM 里的 KV 全部写到 SSD，下次启动能直接从 SSD 恢复
    fn shutdown(&mut self) -> Value {
        let ids: Vec<String> = self.sessions.keys().cloned().collect();
        let t0 = Instant::now();
        let mut n = 0;
        for id in ids {
            let mut s = self.sessions.remove(&id).unwrap();
            if matches!(s.tier, Tier::Gpu | Tier::Dram) {
                match self.to_ssd_sync(&mut s) {
                    Ok(_) => n += 1,
                    Err(e) => eprintln!("[shutdown] {id}: {e}"),
                }
            }
            self.save_meta(&s);
            self.sessions.insert(id, s);
        }
        eprintln!("[shutdown] {n} 个会话的 KV 写到 SSD，用时 {:.2}s", t0.elapsed().as_secs_f64());
        json!({"flushed": n})
    }

    // ------------------------------------------------------------ 进程内测试用

    pub fn cmd_create(&mut self, req: CreateReq) -> Result<Value, ApiErr> {
        self.create(req)
    }
    /// 同步 append，返回 (结果, 每一步的 logits)
    pub fn cmd_append_logits(&mut self, id: &str, req: AppendReq) -> Result<(Value, Vec<Vec<f32>>), ApiErr> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut s = self.sessions.remove(id).ok_or_else(|| not_found(id))?;
        LOG_LOGITS.with(|f| f.set(true));
        LOGIT_LOG.with(|l| l.borrow_mut().clear());
        let r = self.append_inner(&mut s, req, &tx);
        LOG_LOGITS.with(|f| f.set(false));
        self.sessions.insert(id.into(), s);
        r?;
        drop(tx);
        let mut done = Value::Null;
        while let Ok(ev) = rx.try_recv() {
            if let Event::Done(v) = ev {
                done = v;
            }
        }
        Ok((done, LOGIT_LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))))
    }
    pub fn cmd(&mut self, cmd: Cmd) {
        self.handle(cmd)
    }
}

thread_local! {
    /// verify 子命令打开时，记录每一步采样前的完整 logits
    pub static LOGIT_LOG: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
    pub static LOG_LOGITS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// ---------------------------------------------------------------- 采样

fn merge_sampling(req: &Sampling, sess: &Sampling, thinking: bool) -> Sampling {
    // Qwen3 官方推荐：思考模式 0.6/0.95，非思考 0.7/0.8，top_k 20
    let (t, p) = if thinking { (0.6, 0.95) } else { (0.7, 0.8) };
    Sampling {
        temperature: req.temperature.or(sess.temperature).or(Some(t)),
        top_p: req.top_p.or(sess.top_p).or(Some(p)),
        top_k: req.top_k.or(sess.top_k).or(Some(20)),
        presence_penalty: req.presence_penalty.or(sess.presence_penalty).or(Some(0.0)),
        seed: req.seed.or(sess.seed),
        max_tokens: req.max_tokens.or(sess.max_tokens).or(Some(2048)),
    }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(2654435761).wrapping_add(88172645463325252) | 1)
    }
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn sample(logits: &[f32], sp: &Sampling, gen: &[i32], rng: &mut Rng) -> i32 {
    if LOG_LOGITS.with(|f| f.get()) {
        LOGIT_LOG.with(|l| l.borrow_mut().push(logits.to_vec()));
    }
    let temp = sp.temperature.unwrap_or(0.0);
    let pp = sp.presence_penalty.unwrap_or(0.0);
    let penal = |i: usize, x: f32| if pp != 0.0 && gen.contains(&(i as i32)) { x - pp } else { x };
    if temp <= 0.0 {
        let mut best = 0;
        for i in 1..logits.len() {
            if penal(i, logits[i]) > penal(best, logits[best]) {
                best = i;
            }
        }
        return best as i32;
    }
    let k = sp.top_k.unwrap_or(40).clamp(1, logits.len());
    let mut c: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &x)| (i, penal(i, x))).collect();
    c.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    c.truncate(k);
    c.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mx = c[0].1;
    let mut ps: Vec<f32> = c.iter().map(|x| ((x.1 - mx) / temp).exp()).collect();
    let sum: f32 = ps.iter().sum();
    ps.iter_mut().for_each(|p| *p /= sum);
    let top_p = sp.top_p.unwrap_or(1.0);
    let mut acc = 0.0;
    let mut n = ps.len();
    for (i, p) in ps.iter().enumerate() {
        acc += p;
        if acc >= top_p {
            n = i + 1;
            break;
        }
    }
    let total: f32 = ps[..n].iter().sum();
    let mut u = rng.uniform() * total;
    for i in 0..n {
        u -= ps[i];
        if u <= 0.0 {
            return c[i].0 as i32;
        }
    }
    c[n - 1].0 as i32
}

/// 逐 token 流式输出时，攒齐完整的 UTF-8 字符再发
#[derive(Default)]
struct Utf8Stream(Vec<u8>);
impl Utf8Stream {
    fn push(&mut self, b: &[u8]) -> Option<String> {
        self.0.extend_from_slice(b);
        let valid = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(e) if e.error_len().is_some() => self.0.len(), // 真的非法：lossy 输出
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return None;
        }
        let rest = self.0.split_off(valid);
        let out = String::from_utf8_lossy(&self.0).into_owned();
        self.0 = rest;
        Some(out)
    }
    fn flush(&mut self) -> Option<String> {
        if self.0.is_empty() {
            return None;
        }
        Some(String::from_utf8_lossy(&std::mem::take(&mut self.0)).into_owned())
    }
}

/// 拆出思考内容、正文、`<tool_call>` 里的 JSON
pub fn parse_output(text: &str, turn: usize) -> (String, String, Vec<Value>) {
    let mut reasoning = String::new();
    let mut body = text.to_string();
    if let Some(i) = text.find("</think>") {
        reasoning = text[..i].trim_start_matches("<think>").trim().to_string();
        body = text[i + "</think>".len()..].to_string();
    } else if let Some(rest) = text.trim_start().strip_prefix("<think>") {
        // think 没闭合（到了长度上限，或者小模型直接在 think 里发工具调用）：当正文处理
        body = rest.to_string();
    }
    let re = regex::Regex::new(r"(?s)<tool_call>(.*?)(?:</tool_call>|$)").unwrap();
    let mut calls = vec![];
    for (i, cap) in re.captures_iter(&body).enumerate() {
        let raw = cap[1].trim();
        let parsed = parse_call_json(raw);
        let (name, args) = match &parsed {
            Some(v) => (
                v.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                match v.get("arguments") {
                    Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                    Some(a) => a.clone(),
                    None => json!({}),
                },
            ),
            None => (String::new(), json!({})),
        };
        let mut c = json!({"id": format!("call_{turn}_{i}"), "type": "function", "function": {"name": name, "arguments": args}});
        if parsed.is_none() || name.is_empty() {
            c["parse_error"] = json!(raw);
        }
        calls.push(c);
    }
    let content = re.replace_all(&body, "").trim().to_string();
    (reasoning, content, calls)
}

/// 小模型的 JSON 常有多余或缺少的右括号，宽松一点
fn parse_call_json(raw: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        return Some(v);
    }
    let mut s = raw.to_string();
    for _ in 0..3 {
        if s.ends_with('}') {
            s.pop();
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                return Some(v);
            }
        }
    }
    let mut s = raw.to_string();
    for _ in 0..3 {
        s.push('}');
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_calls() {
        let t = "Let me look.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.c\"}}\n</tool_call>";
        let (r, c, calls) = parse_output(t, 1);
        assert_eq!(r, "");
        assert_eq!(c, "Let me look.");
        assert_eq!(calls[0]["function"]["name"], "read_file");
        assert_eq!(calls[0]["function"]["arguments"]["path"], "a.c");
        let (_, _, calls) = parse_output("<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}}\n</tool_call>", 1);
        assert_eq!(calls[0]["function"]["arguments"]["command"], "ls");
    }

    #[test]
    fn utf8_stream() {
        let mut u = Utf8Stream::default();
        let b = "中".as_bytes();
        assert_eq!(u.push(&b[..1]), None);
        assert_eq!(u.push(&b[1..]).as_deref(), Some("中"));
    }
}

#[cfg(test)]
mod tests_think {
    use super::*;

    #[test]
    fn unclosed_think_with_tool_call() {
        let (r, c, calls) = parse_output("<think>\n<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"x\"}}\n</tool_call>", 2);
        assert_eq!((r.as_str(), c.as_str()), ("", ""));
        assert_eq!(calls[0]["function"]["name"], "grep");
        let (r, c, _) = parse_output("<think>\nhmm\n</think>\n\nok", 2);
        assert_eq!((r.as_str(), c.as_str()), ("hmm", "ok"));
    }
}
