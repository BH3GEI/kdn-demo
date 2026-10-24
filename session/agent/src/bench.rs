//! 测量：同一个几千 token 的会话，KV 分别在 GPU / DRAM / SSD / 丢失（重算）时，
//! 下一轮的恢复耗时和首 token 延迟；以及每轮发送字节数 vs 无状态 API 重发全部历史。
//! 全部走 HTTP API，和真实使用路径一致。

use crate::client::Client;
use crate::tools;
use anyhow::Result;
use serde_json::{json, Value};
use std::time::Instant;

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn greedy(msgs: Vec<Value>, max: usize) -> Value {
    json!({"messages": msgs, "temperature": 0.0, "max_tokens": max, "hint": {"waiting": "tool", "tool": "read_file", "expected_s": 1.0}})
}

pub fn run(server: &str, target: usize, reps: usize) -> Result<()> {
    let mut c = Client::new(server);
    let root = std::path::Path::new(REPO).canonicalize()?;
    let defs = tools::definitions();
    let sys = crate::system_prompt(&root);
    let s = c.create(&sys, &defs, Some(false), "bench")?;
    let id = s["id"].as_str().unwrap().to_string();
    let mut messages = vec![json!({"role": "system", "content": sys})];

    // 造一段真实形状的 agent 历史：用户提问 -> 读文件（工具结果）-> 模型短回复，直到 target token
    println!("== 构造会话 {id}（目标 {target} token）==");
    println!("{:>4} {:>8} {:>10} {:>12} {:>8}", "轮", "上下文", "本轮发送", "无状态需发送", "倍数");
    let files = ["src/kvengine.c", "src/common.h", "src/live.c", "src/sim.c", "src/kvengine.h", "src/prefill_bench.c", "README.md"];
    let mut turn = 0;
    let mut pending: Vec<Value> = vec![json!({"role": "user", "content": "Read the engine sources one by one. After each file reply with just OK."})];
    let mut total = s["tokens"].as_u64().unwrap_or(0) as usize;
    let (mut sum_sent, mut sum_stateless) = (0usize, 0usize);
    for f in files.iter().cycle() {
        if total >= target {
            break;
        }
        let stateless = serde_json::to_vec(&json!({"model": "qwen3", "messages": messages.iter().chain(&pending).collect::<Vec<_>>(), "tools": defs, "stream": true}))?.len();
        let (done, sent) = c.append(&id, &greedy(pending.clone(), 8), |_| {})?;
        turn += 1;
        total = done["stats"]["total_tokens"].as_u64().unwrap() as usize;
        sum_sent += sent;
        sum_stateless += stateless;
        println!("{turn:>4} {total:>8} {sent:>10} {stateless:>12} {:>7.1}x", stateless as f64 / sent as f64);
        messages.extend(pending.drain(..));
        messages.push(json!({"role": "assistant", "content": done["content"]}));
        // 下一轮的增量：一个 read_file 的工具结果（模拟本地读文件，每次读 1500 字节左右的一段）
        let text = std::fs::read_to_string(format!("{REPO}/{f}")).unwrap_or_default();
        let text: String = text.chars().take(1500).collect();
        pending = vec![json!({"role": "tool", "name": "read_file", "content": text})];
    }
    println!("合计 {turn} 轮：有状态共发送 {} 字节，无状态需 {} 字节（{:.1} 倍）\n", sum_sent, sum_stateless, sum_stateless as f64 / sum_sent as f64);

    let info = c.get(&format!("/v1/sessions/{id}"))?;
    let base_turns = info["turns"].as_u64().unwrap();
    let ntok = info["tokens"].as_u64().unwrap();

    // 各层恢复：每次先强制放到目标层，再追加同一条短消息，然后回滚
    println!("== 恢复延迟：{ntok} token 的会话，每层 {reps} 次取中位数 ==");
    println!("{:<10} {:>10} {:>10} {:>10} {:>10} {:>11} {:>12}", "KV 位置", "搬运ms", "重算token", "重算ms", "新增prefill", "TTFT(服务端)", "TTFT(客户端)");
    let probe = vec![json!({"role": "user", "content": "Which file defines ke_restore? Answer with the file name only."})];
    let mut kv_bytes = 0u64;
    for tier in ["gpu", "dram", "ssd", "drop"] {
        let (mut rest, mut rec, mut pf, mut ttft, mut cttft) = (vec![], vec![], vec![], vec![], vec![]);
        let mut rec_tok = 0;
        for _ in 0..reps {
            let d = c.post(&format!("/v1/admin/sessions/{id}/demote"), &json!({"tier": tier}))?;
            if tier == "ssd" {
                kv_bytes = d["bytes"].as_u64().unwrap_or(0);
            }
            let t0 = Instant::now();
            let mut first = None;
            let (done, _) = c.append(&id, &greedy(probe.clone(), 4), |_| {
                first.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
            })?;
            let st = &done["stats"];
            rest.push(st["restore_ms"].as_f64().unwrap());
            rec.push(st["recompute_ms"].as_f64().unwrap());
            pf.push(st["prefill_ms"].as_f64().unwrap());
            ttft.push(st["ttft_ms"].as_f64().unwrap());
            cttft.push(first.unwrap_or(f64::NAN));
            rec_tok = st["recompute_tokens"].as_u64().unwrap();
            c.post(&format!("/v1/sessions/{id}/rollback"), &json!({"turn": base_turns}))?;
        }
        let name = if tier == "drop" { "重算" } else { tier };
        println!(
            "{:<10} {:>10.1} {:>10} {:>10.1} {:>10.1} {:>11.1} {:>12.1}",
            name,
            median(&mut rest),
            rec_tok,
            median(&mut rec),
            median(&mut pf),
            median(&mut ttft),
            median(&mut cttft),
        );
    }
    println!("SSD 上的 KV 文件 {:.0}MB（{:.0}KB/token）", kv_bytes as f64 / 1e6, kv_bytes as f64 / 1024.0 / ntok as f64);
    println!("（重算时系统提示 + 工具定义命中共享前缀缓存，不在重算 token 里）");
    c.delete(&format!("/v1/sessions/{id}"))?;
    Ok(())
}
