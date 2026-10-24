//! kdn-agent：在任意本地目录里跑的编码 Agent。
//!
//! 会话在服务端（kdn-server）持有 KV；这里每轮只发增量（用户消息或工具结果），
//! 并自动发等待提示：工具调用之后告诉服务端在等哪个工具、大概多久；
//! 要人确认或一轮结束时告诉它在等人；退出时告诉它 done。

mod bench;
mod client;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use client::Client;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use tools::Tools;

#[derive(Parser)]
#[command(about = "本地编码 Agent，连有状态会话服务（只发增量）")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Sub>,
    #[arg(long, default_value = "http://127.0.0.1:8399", global = true)]
    server: String,
    /// 自动同意 bash / write_file / edit_file
    #[arg(long)]
    yes: bool,
    /// 恢复指定会话
    #[arg(long)]
    resume: Option<String>,
    /// 恢复本目录上次的会话
    #[arg(long, short = 'c')]
    r#continue: bool,
    /// 只跑这一条提示（含工具循环）然后退出
    #[arg(long, short = 'p')]
    prompt: Option<String>,
    /// 打开 Qwen3 思考模式（慢，但工具调用更靠谱）
    #[arg(long)]
    thinking: bool,
    /// 一次用户提问里最多几轮工具调用
    #[arg(long, default_value_t = 16)]
    max_steps: usize,
    #[arg(long)]
    temperature: Option<f32>,
}

#[derive(Subcommand)]
enum Sub {
    /// 恢复延迟（GPU / DRAM / SSD / 重算）和每轮字节数测量
    Bench {
        #[arg(long, default_value_t = 3000)]
        tokens: usize,
        #[arg(long, default_value_t = 5)]
        reps: usize,
    },
}

fn system_prompt(root: &std::path::Path) -> String {
    format!(
        "You are kdn-agent, a coding assistant working inside the user's local repository at {}.\n\
         Use the tools to look at files and run commands; paths are relative to the repository root.\n\
         Work step by step: list or read the relevant files, run the program or tests to see the problem, \
         then explain the cause and propose a concrete fix (apply it with edit_file when the user asks).\n\
         Before editing, state the root cause in one sentence. Call one tool at a time and wait for its result.",
        root.display()
    )
}

/// 每个目录上次用的会话，存在 ~/.kdn-agent/state.json
fn state_path() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".kdn-agent/state.json")
}
fn load_last(root: &std::path::Path) -> Option<String> {
    let v: Value = serde_json::from_slice(&std::fs::read(state_path()).ok()?).ok()?;
    v.get(root.to_string_lossy().as_ref())?.as_str().map(String::from)
}
fn save_last(root: &std::path::Path, id: &str) {
    let p = state_path();
    let mut v: Value = std::fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
    v[root.to_string_lossy().as_ref()] = json!(id);
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let _ = std::fs::write(&p, serde_json::to_vec_pretty(&v).unwrap());
}

fn kb(n: usize) -> String {
    if n < 1024 {
        format!("{n}B")
    } else {
        format!("{:.1}KB", n as f64 / 1024.0)
    }
}

/// 每轮统计
#[derive(Default)]
struct Totals {
    turns: usize,
    sent: usize,
    stateless: usize,
}

struct Agent {
    c: Client,
    tools: Tools,
    id: String,
    /// 本地镜像的对话（只用来算"无状态 API 每轮要重发多少字节"和展示）
    messages: Vec<Value>,
    tool_defs: Vec<Value>,
    auto_yes: bool,
    thinking: bool,
    max_steps: usize,
    temperature: Option<f32>,
    totals: Totals,
}

const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const YEL: &str = "\x1b[33m";
const RST: &str = "\x1b[0m";

impl Agent {
    fn new_session(&mut self) -> Result<()> {
        let sys = system_prompt(&self.tools.root);
        let v = self.c.create(&sys, &self.tool_defs, Some(self.thinking), &self.tools.root.display().to_string())?;
        self.id = v["id"].as_str().unwrap().to_string();
        self.messages = vec![json!({"role": "system", "content": sys})];
        save_last(&self.tools.root, &self.id);
        println!("{DIM}新会话 {}：系统提示 {} token，prefill {}ms（共享前缀命中 {} token）{RST}", self.id, v["stats"]["system_tokens"], v["stats"]["prefill_ms"], v["stats"]["prefix_hit_tokens"]);
        Ok(())
    }

    fn resume(&mut self, id: &str) -> Result<()> {
        let v = self.c.get(&format!("/v1/sessions/{id}?messages=true"))?;
        self.id = id.to_string();
        self.messages = v["messages"].as_array().cloned().unwrap_or_default();
        self.tool_defs = v["tools"].as_array().cloned().unwrap_or_default();
        save_last(&self.tools.root, id);
        println!("{DIM}恢复会话 {id}：{} token，{} 轮，KV 当前在 {}{RST}", v["tokens"], v["turns"], v["tier"]);
        for m in self.messages.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev() {
            let role = m["role"].as_str().unwrap_or("");
            if role == "system" {
                continue;
            }
            let mut s: String = m["content"].as_str().unwrap_or("").chars().take(160).collect();
            if let Some(tc) = m["tool_calls"].as_array() {
                for c in tc {
                    s += &format!(" -> {}({})", c["function"]["name"].as_str().unwrap_or(""), c["function"]["arguments"]);
                }
            }
            println!("{DIM}  [{role}] {}{RST}", s.replace('\n', " "));
        }
        Ok(())
    }

    /// 无状态 API 这一轮要发的请求体：完整历史 + 工具定义
    fn stateless_bytes(&self, new: &[Value]) -> usize {
        let mut all = self.messages.clone();
        all.extend(new.iter().cloned());
        serde_json::to_vec(&json!({"model": "qwen3", "messages": all, "tools": self.tool_defs, "stream": true})).unwrap().len()
    }

    /// 发一次增量，流式打印；返回 done
    fn step(&mut self, new: Vec<Value>) -> Result<Value> {
        let stateless = self.stateless_bytes(&new);
        let mut body = json!({"messages": new});
        if let Some(t) = self.temperature {
            body["temperature"] = json!(t);
        }
        // 流式打印，但把 <tool_call> 原文藏起来，结束后再好好显示
        let mut full = String::new();
        let mut printed = 0;
        let mut in_think = false;
        let (done, sent) = self.c.append(&self.id, &body, |t| {
            full.push_str(t);
            let visible = match full.find("<tool_call>") {
                Some(i) => i,
                None => {
                    // 末尾可能是 "<tool_c" 这样的半截，先别打
                    let tail = (1.."<tool_call>".len()).rev().find(|&k| full.ends_with(&"<tool_call>"[..k])).unwrap_or(0);
                    full.len() - tail
                }
            };
            if visible > printed && full.is_char_boundary(visible) {
                let s = &full[printed..visible];
                if s.contains("<think>") {
                    in_think = true;
                }
                print!("{}{s}", if in_think { DIM } else { "" });
                if s.contains("</think>") {
                    in_think = false;
                    print!("{RST}");
                }
                let _ = std::io::stdout().flush();
                printed = visible;
            }
        })?;
        println!("{RST}");
        self.messages.extend(new);
        let mut am = json!({"role": "assistant", "content": done["content"]});
        if done["tool_calls"].as_array().is_some_and(|a| !a.is_empty()) {
            am["tool_calls"] = done["tool_calls"].clone();
        }
        self.messages.push(am);
        let st = &done["stats"];
        let tier = match st["kv_tier"].as_str().unwrap_or("") {
            "gpu" => "GPU".to_string(),
            "dram" => format!("DRAM（恢复 {}ms）", st["restore_ms"]),
            "ssd" => format!("SSD（恢复 {}ms）", st["restore_ms"]),
            "recompute" | "ssd-failed" => format!("重算 {} token（{}ms）", st["recompute_tokens"], st["recompute_ms"]),
            t => t.to_string(),
        };
        let extra = if st["recompute_tokens"].as_u64().unwrap_or(0) > 0 && st["kv_tier"] != "recompute" { format!(" + 重算 {} token", st["recompute_tokens"]) } else { String::new() };
        println!(
            "{DIM}[KV 在 {tier}{extra} | 新增 {} token，prefill {}ms | TTFT {}ms | 生成 {} token，{} tok/s | 发送 {}（无状态需 {}）| 上下文 {} token]{RST}",
            st["new_tokens"], st["prefill_ms"], st["ttft_ms"], st["gen_tokens"], st["decode_tok_s"], kb(sent), kb(stateless), st["total_tokens"]
        );
        self.totals.turns += 1;
        self.totals.sent += sent;
        self.totals.stateless += stateless;
        Ok(done)
    }

    /// 一次用户提问：循环执行工具直到模型不再调用
    fn user_turn(&mut self, text: &str) -> Result<()> {
        let mut new = vec![json!({"role": "user", "content": text})];
        let mut last_calls: Vec<String> = vec![];
        for _ in 0..self.max_steps {
            let done = self.step(new)?;
            let calls = done["tool_calls"].as_array().cloned().unwrap_or_default();
            if calls.is_empty() {
                // 一轮结束：等人
                let h = self.c.hint(&self.id, "human", None, None)?;
                println!("{DIM}[hint: 等人回复 -> {}]{RST}", h["placement"].as_str().unwrap_or(""));
                return Ok(());
            }
            new = vec![];
            for c in &calls {
                let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                let args = c["function"]["arguments"].clone();
                println!("{CYAN}-> {name}({}){RST}", serde_json::to_string(&args).unwrap());
                let sig = format!("{name}{args}");
                last_calls.push(sig.clone());
                let result = if name.is_empty() {
                    format!("Error: could not parse tool call: {}", c["parse_error"].as_str().unwrap_or(""))
                } else if last_calls.iter().rev().take(3).filter(|s| **s == sig).count() >= 3 {
                    "Error: you already made this exact call 3 times. Use the earlier result or do something else.".to_string()
                } else {
                    self.run_tool(&name, &args)?
                };
                let preview: String = result.lines().take(6).collect::<Vec<_>>().join("\n");
                let more = result.lines().count().saturating_sub(6);
                println!("{DIM}{preview}{}{RST}", if more > 0 { format!("\n  ... ({more} more lines)") } else { String::new() });
                new.push(json!({"role": "tool", "tool_call_id": c["id"], "name": name, "content": result}));
            }
        }
        println!("{YEL}[达到每次提问的最大工具轮数 {}，停下]{RST}", self.max_steps);
        let _ = self.c.hint(&self.id, "human", None, None);
        Ok(())
    }

    fn run_tool(&mut self, name: &str, args: &Value) -> Result<String> {
        if tools::needs_confirm(name) && !self.auto_yes {
            // 等用户确认的这段时间，瓶颈是人
            self.c.hint(&self.id, "human", None, None)?;
            let what = match name {
                "bash" => format!("运行命令：{}", args["command"].as_str().unwrap_or("")),
                "write_file" => format!("写文件 {}（{} 字节）", args["path"].as_str().unwrap_or(""), args["content"].as_str().map_or(0, str::len)),
                _ => format!("编辑 {}：\n{YEL}- {}\n+ {}{RST}\n", args["path"].as_str().unwrap_or(""), args["old_string"].as_str().unwrap_or("").replace('\n', "\n- "), args["new_string"].as_str().unwrap_or("").replace('\n', "\n+ ")),
            };
            let (ok, always) = tools::confirm(&format!("{YEL}{what}{RST}\n允许？"));
            if always {
                self.auto_yes = true;
            }
            if !ok {
                return Ok("The user denied this action.".into());
            }
        }
        let exp = tools::expected_s(name, args);
        let h = self.c.hint(&self.id, "tool", Some(name), exp)?;
        print!("{DIM}[hint: 等 {name}，预计 {:.1}s -> {}]{RST} ", h["predicted_gap_s"].as_f64().unwrap_or(0.0), h["placement"].as_str().unwrap_or(""));
        let _ = std::io::stdout().flush();
        let t0 = std::time::Instant::now();
        let r = self.tools.run(name, args);
        println!("{DIM}({:.2}s){RST}", t0.elapsed().as_secs_f64());
        Ok(r)
    }

    fn command(&mut self, line: &str) -> Result<bool> {
        let mut it = line.split_whitespace();
        let cmd = it.next().unwrap_or("");
        let arg = it.next();
        match cmd {
            "/quit" | "/exit" => {
                let h = self.c.hint(&self.id, "done", None, None)?;
                println!("{DIM}[hint: done -> {}]{RST}", h["placement"].as_str().unwrap_or(""));
                return Ok(false);
            }
            "/sessions" => {
                let v = self.c.get("/v1/sessions")?;
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
                println!("{:<11} {:>7} {:>5} {:<20} {:>9}  {}", "id", "token", "轮", "KV 位置", "上次活动", "标签");
                for s in v["sessions"].as_array().unwrap() {
                    let ago = now - s["updated"].as_f64().unwrap_or(now);
                    let ago = if ago < 120.0 { format!("{ago:.0}s 前") } else if ago < 7200.0 { format!("{:.0}m 前", ago / 60.0) } else { format!("{:.1}h 前", ago / 3600.0) };
                    let mark = if s["id"] == self.id.as_str() { "*" } else { " " };
                    println!("{mark}{:<10} {:>7} {:>5} {:<20} {:>9}  {}", s["id"].as_str().unwrap_or(""), s["tokens"], s["turns"], s["tier"].as_str().unwrap_or(""), ago, s["label"].as_str().unwrap_or(""));
                }
            }
            "/resume" => match arg {
                Some(id) => self.resume(id)?,
                None => println!("用法：/resume <id>"),
            },
            "/new" => self.new_session()?,
            "/fork" => {
                let v = self.c.post(&format!("/v1/sessions/{}/fork", self.id), &json!({}))?;
                let child = v["id"].as_str().unwrap().to_string();
                println!("{DIM}fork {} -> {child}：共享 {} token 的 KV 前缀（seq_cp），用时 {}ms；已切到新会话{RST}", self.id, v["stats"]["shared_tokens"], v["stats"]["fork_ms"]);
                self.resume(&child)?;
            }
            "/rollback" => match arg.and_then(|a| a.parse::<usize>().ok()) {
                Some(n) => {
                    let v = self.c.post(&format!("/v1/sessions/{}/rollback", self.id), &json!({"turn": n}))?;
                    println!("{DIM}回滚到第 {n} 轮：删掉 {} token，剩 {} token{RST}", v["removed_tokens"], v["tokens"]);
                    let id = self.id.clone();
                    self.resume(&id)?;
                }
                None => println!("用法：/rollback N（保留前 N 轮；每次 append 算一轮）"),
            },
            "/stats" => {
                let s = self.c.get(&format!("/v1/sessions/{}", self.id))?;
                println!("会话 {}：{} token，{} 轮，KV 在 {}，SSD 文件覆盖 {} token", self.id, s["tokens"], s["turns"], s["tier"], s["ssd_tokens"]);
                println!("等待：{}", s["waiting"]);
                let t = &self.totals;
                if t.turns > 0 {
                    println!("本进程 {} 次请求：共发送 {}，无状态 API 需 {}（{:.1} 倍）", t.turns, kb(t.sent), kb(t.stateless), t.stateless as f64 / t.sent.max(1) as f64);
                }
                println!("服务端：{}", self.c.get("/v1/admin/stats")?);
            }
            "/demote" => {
                let v = self.c.post(&format!("/v1/admin/sessions/{}/demote", self.id), &json!({"tier": arg.unwrap_or("ssd")}))?;
                println!("{DIM}{v}{RST}");
            }
            "/help" => println!("/sessions /resume <id> /new /fork /rollback N /stats /demote <dram|ssd|drop> /quit"),
            _ => println!("未知命令，/help 看帮助"),
        }
        Ok(true)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(Sub::Bench { tokens, reps }) = cli.cmd {
        return bench::run(&cli.server, tokens, reps);
    }
    let root = std::env::current_dir()?.canonicalize()?;
    let mut a = Agent {
        c: Client::new(&cli.server),
        tools: Tools { root: root.clone() },
        id: String::new(),
        messages: vec![],
        tool_defs: tools::definitions(),
        auto_yes: cli.yes,
        thinking: cli.thinking,
        max_steps: cli.max_steps,
        temperature: cli.temperature,
        totals: Totals::default(),
    };
    let resume = cli.resume.clone().or_else(|| if cli.r#continue { load_last(&root) } else { None });
    match resume {
        Some(id) => a.resume(&id)?,
        None => a.new_session()?,
    }
    if let Some(p) = &cli.prompt {
        println!("{CYAN}> {p}{RST}");
        a.user_turn(p)?;
        let t = &a.totals;
        println!("{DIM}[合计 {} 次请求：发送 {}，无状态 API 需 {}]{RST}", t.turns, kb(t.sent), kb(t.stateless));
        return Ok(());
    }
    println!("{DIM}工作目录 {}；/help 看命令{RST}", root.display());
    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            Err(_) => {
                let _ = a.c.hint(&a.id, "done", None, None);
                break;
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line);
        let r = if line.starts_with('/') { a.command(line) } else { a.user_turn(line).map(|_| true) };
        match r {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => println!("{YEL}错误：{e:#}{RST}"),
        }
    }
    Ok(())
}
