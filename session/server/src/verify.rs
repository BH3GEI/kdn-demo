//! 两个自检：
//!   check-template：增量拼接 vs GGUF 里真实的 Jinja 模板整段渲染（文本和 token 都比）
//!   verify        ：同一个 append，KV 分别来自 GPU / DRAM / SSD / 从 token 历史重算，
//!                   贪心解码下比较输出 token 和每一步完整 logits 的最大绝对差

use crate::engine::{AppendReq, ApiErr, Cmd, Config, CreateReq, Engine, Sampling};
use crate::llama::Model;
use crate::template::{self, EMPTY_THINK};
use serde_json::{json, Value};

pub fn sample_tools() -> Vec<Value> {
    vec![
        json!({"type": "function", "function": {"name": "read_file", "description": "Read a text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}),
        json!({"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}),
    ]
}

/// 去掉除最后一个以外的空 think 块（Qwen3 模板重渲染历史时会删掉它们）
fn strip_old_empty_think(s: &str) -> (String, usize) {
    let n = s.matches(EMPTY_THINK).count();
    if n <= 1 {
        return (s.to_string(), 0);
    }
    let last = s.rfind(EMPTY_THINK).unwrap();
    (s[..last].replace(EMPTY_THINK, "") + &s[last..], n - 1)
}

struct Step {
    new_msgs: Vec<Value>,
    /// 模拟模型生成的这一轮（assistant 消息）
    reply: Value,
}

fn call(name: &str, args: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "arguments": args}})
}

/// 按我们的增量方式拼出整段，逐步和 Jinja 渲染比较；返回（不一致步数，删掉的旧 think 块数）
fn run_scenario(model: &Model, tmpl: &str, title: &str, thinking: bool, steps: &[Step]) -> (usize, usize) {
    let system = "You are a coding agent.";
    let tools = sample_tools();
    let mut msgs = vec![json!({"role": "system", "content": system})];
    let sys = template::system_text(system, &tools);
    let mut ours = sys.clone();
    let mut pieces_tok: Vec<i32> = model.tokenize(&sys, model.add_bos());
    let mut pending = false;
    let mut bad = 0;
    let mut stripped = 0;
    println!("\n== {title} ==");
    for (k, st) in steps.iter().enumerate() {
        let d = template::delta_text(pending, &st.new_msgs, thinking).unwrap();
        ours += &d;
        pieces_tok.extend(model.tokenize(&d, false));
        msgs.extend(st.new_msgs.iter().cloned());
        // 此时（生成提示之后）应当和模板 add_generation_prompt=true 的整段渲染一致
        let full = template::render_jinja(tmpl, &Value::Array(msgs.clone()), &json!(tools), true, thinking).expect("Jinja 渲染失败");
        let (norm, n) = strip_old_empty_think(&ours);
        let ok = norm == full;
        println!("  第 {} 次 append：增量 {:>4} 字节，整段 {:>5} 字节 → {}{}", k + 1, d.len(), full.len(), if ok { "一致" } else { "不一致" }, if n > 0 { format!("（模板删掉了 {n} 个历史空 think 块）") } else { String::new() });
        if !ok {
            bad += 1;
            let i = norm.bytes().zip(full.bytes()).take_while(|(a, b)| a == b).count();
            println!("    分歧位置 {i}：\n    ours: {:?}\n    tmpl: {:?}", &norm[i.saturating_sub(40)..(i + 80).min(norm.len())], &full[i.saturating_sub(40)..(i + 80).min(full.len())]);
        }
        stripped = n;
        // 模拟模型生成：思考内容 + 正文 + 工具调用
        let r = &st.reply;
        let content = r["content"].as_str().unwrap_or("");
        let calls: Vec<(String, Value)> = r["tool_calls"].as_array().map(|a| a.iter().map(|c| (c["function"]["name"].as_str().unwrap().to_string(), c["function"]["arguments"].clone())).collect()).unwrap_or_default();
        let mut body = template::assistant_body(content, &calls);
        if let Some(rc) = r.get("reasoning_content").and_then(Value::as_str) {
            body = format!("<think>\n{rc}\n</think>\n\n{body}");
        }
        ours += &body;
        pieces_tok.extend(model.tokenize(&body, false));
        msgs.push(r.clone());
        pending = true;
    }
    // 最后补上 <|im_end|>\n，和 add_generation_prompt=false 的整段比
    ours += "<|im_end|>\n";
    pieces_tok.extend(model.tokenize("<|im_end|>\n", false));
    let full = template::render_jinja(tmpl, &Value::Array(msgs.clone()), &json!(tools), false, thinking).unwrap();
    let (norm, n) = strip_old_empty_think(&ours);
    println!("  结束后整段：{}{}", if norm == full { "一致" } else { "不一致" }, if n > 0 { format!("（模板删掉了 {n} 个历史空 think 块）") } else { String::new() });
    if norm != full {
        bad += 1;
    }
    // 分段分词 == 整段分词？（每段都从特殊 token 开始，边界不会合并）
    let whole = model.tokenize(&ours, model.add_bos());
    println!("  分段分词 {} token，整段分词 {} token → {}", pieces_tok.len(), whole.len(), if whole == pieces_tok { "逐 token 一致" } else { "不一致" });
    if whole != pieces_tok {
        bad += 1;
    }
    (bad, stripped.max(n))
}

pub fn check_template(cfg: &Config) -> anyhow::Result<()> {
    let model = Model::load(&cfg.model_path, 0)?;
    let tmpl = model.meta("tokenizer.chat_template").ok_or_else(|| anyhow::anyhow!("GGUF 里没有 tokenizer.chat_template"))?;
    println!("模型 {}，chat_template {} 字节（从 GGUF 元数据读出）", cfg.model_path, tmpl.len());
    let user = |c: &str| json!({"role": "user", "content": c});
    let tool = |c: &str| json!({"role": "tool", "content": c});
    // A：非思考模式，多轮、多工具结果、多次真实用户提问
    let a = [
        Step { new_msgs: vec![user("Fix the bug in main.c")], reply: json!({"role": "assistant", "content": "", "tool_calls": [call("read_file", json!({"path": "main.c"}))]}) },
        Step { new_msgs: vec![tool("int main() { return 1; }"), tool("second result")], reply: json!({"role": "assistant", "content": "Let me run it.", "tool_calls": [call("bash", json!({"command": "cc main.c && ./a.out; echo $?"})), call("read_file", json!({"path": "中文.txt"}))]}) },
        Step { new_msgs: vec![tool("1"), tool("文件内容")], reply: json!({"role": "assistant", "content": "main returns 1; change it to 0."}) },
        Step { new_msgs: vec![user("ok, do it")], reply: json!({"role": "assistant", "content": "", "tool_calls": [call("bash", json!({"command": "sed -i '' 's/return 1/return 0/' main.c"}))]}) },
        Step { new_msgs: vec![tool("")], reply: json!({"role": "assistant", "content": "Done."}) },
    ];
    let (b1, s1) = run_scenario(&model, &tmpl, "场景 A：非思考模式，5 次 append（2 次用户提问 + 3 次工具结果）", false, &a);
    // B：思考模式，一次提问内的工具循环（模板保留这一段的思考内容）
    let b = [
        Step { new_msgs: vec![user("What is in main.c?")], reply: json!({"role": "assistant", "content": "", "reasoning_content": "I should read the file.", "tool_calls": [call("read_file", json!({"path": "main.c"}))]}) },
        Step { new_msgs: vec![tool("int main() { return 1; }")], reply: json!({"role": "assistant", "content": "It returns 1.", "reasoning_content": "The file is short."}) },
    ];
    let (b2, _) = run_scenario(&model, &tmpl, "场景 B：思考模式，一次提问内的工具循环", true, &b);
    println!(
        "\n结论：{}。每次 append 的增量 = 模板整段渲染的对应后缀；唯一差别是 Qwen3 模板重渲染时删掉历史 assistant 轮的空 think 块（场景 A 删了 {s1} 个），\
         KV 里保留模型生成时真实看到的 token，不改写历史。",
        if b1 + b2 == 0 { "全部一致" } else { "有不一致，见上" }
    );
    anyhow::ensure!(b1 + b2 == 0, "模板核对失败");
    Ok(())
}

// ---------------------------------------------------------------- 恢复 vs 重算

fn sync(e: &mut Engine, f: impl FnOnce(tokio::sync::oneshot::Sender<Result<Value, ApiErr>>) -> Cmd) -> Value {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    e.cmd(f(tx));
    match rx.try_recv().expect("引擎没有回复") {
        Ok(v) => v,
        Err(ApiErr(c, m)) => panic!("{c}: {m}"),
    }
}

fn greedy(msg: &str, max: usize) -> AppendReq {
    AppendReq {
        messages: vec![json!({"role": "user", "content": msg})],
        sampling: Sampling { temperature: Some(0.0), max_tokens: Some(max), ..Default::default() },
        hint: Some(crate::engine::HintReq { waiting: "tool".into(), tool: Some("read_file".into()), expected_s: Some(1.0) }),
        debug: true,
        request_bytes: 0,
    }
}

pub fn restore_vs_recompute(mut cfg: Config, target_tokens: usize, gen: usize) -> anyhow::Result<()> {
    cfg.prefix_cache = 0; // 重算要从位置 0 真算，不借共享前缀
    cfg.data_dir = std::env::temp_dir().join(format!("kdn-verify-{}", std::process::id()));
    cfg.human_idle = 1e9;
    let (tx, _rx) = std::sync::mpsc::channel();
    let mut e = Engine::new(cfg.clone(), tx)?;
    println!("模型 {}", e.model_id);

    // 历史：系统提示 + 一大段源码 + 两轮问答（真实生成）
    let src = ["src/kvengine.c", "src/common.h", "src/kvengine.h", "src/live.c"]
        .iter()
        .filter_map(|f| std::fs::read_to_string(format!("{}/../../{f}", env!("CARGO_MANIFEST_DIR"))).ok())
        .collect::<Vec<_>>()
        .join("\n");
    let mut doc = String::new();
    for line in src.lines() {
        if e.model.tokenize(&doc, false).len() >= target_tokens.saturating_sub(300) {
            break;
        }
        doc.push_str(line);
        doc.push('\n');
    }
    let created = e.cmd_create(CreateReq { system: "You are a careful C code reviewer.".into(), tools: sample_tools(), ..Default::default() }).map_err(|e| anyhow::anyhow!(e.1))?;
    let id = created["id"].as_str().unwrap().to_string();
    e.cmd_append_logits(&id, greedy(&format!("Here is some code:\n```c\n{doc}```\nSummarize what it does in two sentences."), 64)).map_err(|e| anyhow::anyhow!(e.1))?;
    e.cmd_append_logits(&id, greedy("Which function moves KV from SSD back to GPU?", 48)).map_err(|e| anyhow::anyhow!(e.1))?;
    let turns = sync(&mut e, |r| Cmd::Get(id.clone(), false, r))["turns"].as_u64().unwrap() as usize;
    let hist = sync(&mut e, |r| Cmd::Get(id.clone(), false, r))["tokens"].as_u64().unwrap();
    println!("历史 {hist} token，{turns} 轮；测试 append 贪心生成 {gen} token\n");

    let probe = "List the three KV tiers in this engine, one per line.";
    let mut base: Option<(Vec<i64>, Vec<Vec<f32>>)> = None;
    println!("{:<10} {:>8} {:>10} {:>10} {:>9} {:>10}  {}", "KV 来源", "恢复ms", "重算token", "重算ms", "TTFT ms", "token一致", "logits 最大绝对差（逐步）");
    for tier in ["gpu", "dram", "ssd", "drop"] {
        if tier != "gpu" {
            sync(&mut e, |r| Cmd::Demote(id.clone(), tier.into(), r));
        }
        let (done, logits) = e.cmd_append_logits(&id, greedy(probe, gen)).map_err(|e| anyhow::anyhow!(e.1))?;
        let toks: Vec<i64> = done["token_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let st = &done["stats"];
        let (same, diff) = match &base {
            None => (true, 0.0f32),
            Some((bt, bl)) => {
                // 只比较两边 token 前缀一致的那些步（之后的输入不同，logits 没有可比性）
                let agree = bt.iter().zip(&toks).take_while(|(a, b)| a == b).count();
                let steps = (agree + 1).min(bl.len()).min(logits.len());
                let d = (0..steps).flat_map(|i| bl[i].iter().zip(&logits[i]).map(|(a, b)| (a - b).abs())).fold(0.0f32, f32::max);
                (bt == &toks, d)
            }
        };
        println!(
            "{:<10} {:>8} {:>10} {:>10} {:>9} {:>10}  {:.6}",
            if tier == "drop" { "recompute" } else { tier },
            st["restore_ms"],
            st["recompute_tokens"],
            st["recompute_ms"],
            st["ttft_ms"],
            if same { "是" } else { "否" },
            diff
        );
        if base.is_none() {
            println!("           输出：{:?}", done["content"].as_str().unwrap_or(""));
            base = Some((toks, logits));
        }
        sync(&mut e, |r| Cmd::Rollback(id.clone(), turns, r));
    }
    let _ = std::fs::remove_dir_all(&cfg.data_dir);
    Ok(())
}
