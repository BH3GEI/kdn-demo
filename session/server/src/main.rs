//! kdn-server：有状态会话推理服务。
//!
//!   kdn-server [serve]          起 HTTP 服务（默认 127.0.0.1:8399）
//!   kdn-server check-template   用 GGUF 里真实的 Jinja 模板核对增量拼接
//!   kdn-server verify           SSD 恢复 vs 从 token 历史重算：输出 token 和 logits 对比

mod api;
mod engine;
mod kvfile;
mod llama;
mod policy;
mod template;
mod verify;

use clap::{Parser, Subcommand};
use engine::{Cmd, Config, Engine};
use std::path::PathBuf;

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

#[derive(Parser)]
#[command(about = "有状态会话 LLM 推理服务（llama.cpp / Metal）")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Sub>,
    /// GGUF 模型；默认 kdn-demo/models/qwen3-0.6b-q8_0.gguf
    #[arg(long, global = true)]
    model: Option<String>,
    /// 会话元数据和 SSD 层 KV 文件目录；默认 kdn-demo/data/sessions
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8399", global = true)]
    addr: String,
    /// llama.cpp 上下文（所有 GPU 会话共用的 KV 单元数，kv_unified）
    #[arg(long, default_value_t = 32768, global = true)]
    n_ctx: usize,
    #[arg(long, default_value_t = 1024, global = true)]
    n_batch: usize,
    /// 最多同时在 GPU 上的序列数（含共享前缀）
    #[arg(long, default_value_t = 32, global = true)]
    n_seq: usize,
    /// GPU 层容量（token）；默认 n_ctx 的 3/4，其余留给生成
    #[arg(long, global = true)]
    gpu_tokens: Option<usize>,
    /// DRAM 层容量（token），Qwen3 0.6B/1.7B 的 KV 约 112KB/token
    #[arg(long, default_value_t = 65536, global = true)]
    dram_tokens: usize,
    /// 预测空档超过它（秒）直接降到 SSD
    #[arg(long, default_value_t = 1800.0, global = true)]
    cold_gap: f64,
    /// 等人回复时，GPU 上空闲多久（秒）后降到 DRAM
    #[arg(long, default_value_t = 10.0, global = true)]
    human_idle: f64,
    /// DRAM 里空闲多久（秒）后写到 SSD
    #[arg(long, default_value_t = 600.0, global = true)]
    dram_idle: f64,
    /// 默认打开 Qwen3 思考模式（默认关，快）
    #[arg(long, global = true)]
    thinking: bool,
    /// 共享前缀缓存（系统提示 + 工具定义）条数，0 关闭
    #[arg(long, default_value_t = 4, global = true)]
    prefix_cache: usize,
}

#[derive(Subcommand)]
enum Sub {
    Serve,
    CheckTemplate,
    Verify {
        /// 历史长度（大约多少 token）
        #[arg(long, default_value_t = 3000)]
        tokens: usize,
        #[arg(long, default_value_t = 48)]
        gen: usize,
    },
}

fn config(c: &Cli) -> Config {
    Config {
        model_path: c.model.clone().unwrap_or_else(|| format!("{REPO}/models/qwen3-0.6b-q8_0.gguf")),
        data_dir: c.data_dir.clone().unwrap_or_else(|| PathBuf::from(format!("{REPO}/data/sessions"))),
        n_ctx: c.n_ctx,
        n_batch: c.n_batch,
        n_seq: c.n_seq,
        gpu_tokens: c.gpu_tokens.unwrap_or(c.n_ctx * 3 / 4),
        dram_tokens: c.dram_tokens,
        cold_gap: c.cold_gap,
        human_idle: c.human_idle,
        dram_idle: c.dram_idle,
        thinking: c.thinking,
        prefix_cache: c.prefix_cache,
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = config(&cli);
    match cli.cmd {
        Some(Sub::CheckTemplate) => verify::check_template(&cfg),
        Some(Sub::Verify { tokens, gen }) => verify::restore_vs_recompute(cfg, tokens, gen),
        None | Some(Sub::Serve) => serve(cfg, &cli.addr),
    }
}

fn serve(cfg: Config, addr: &str) -> anyhow::Result<()> {
    eprintln!("[serve] 模型 {}", cfg.model_path);
    let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
    // 引擎线程：独占 llama context。先在线程里建好，失败直接退出
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let etx = tx.clone();
    let engine_thread = std::thread::spawn(move || match Engine::new(cfg, etx) {
        Ok(e) => {
            let _ = ready_tx.send(Ok(e.model_id.clone()));
            e.run(rx);
        }
        Err(err) => {
            let _ = ready_tx.send(Err(err.to_string()));
        }
    });
    let model_id = ready_rx.recv()?.map_err(anyhow::Error::msg)?;
    eprintln!("[serve] 引擎就绪：{model_id}");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("[serve] 监听 http://{addr}");
        let st = api::AppState { tx: tx.clone() };
        let shutdown = async move {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("\n[serve] 收到退出信号，把 KV 落盘……");
            let (r, w) = tokio::sync::oneshot::channel();
            let _ = tx.send(Cmd::Shutdown(r));
            let _ = w.await;
        };
        axum::serve(listener, api::router(st)).with_graceful_shutdown(shutdown).await?;
        anyhow::Ok(())
    })?;
    let _ = engine_thread.join();
    Ok(())
}
