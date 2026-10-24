//! Qwen3 的 ChatML / Hermes 工具调用格式，按"只追加"的方式拼。
//!
//! 和 GGUF 里 tokenizer.chat_template（Jinja）逐字节一致的片段：
//!   系统提示（含 `# Tools` 和 `<tools>` 列表）、user 消息、连续的 tool 结果
//!   （合在一个 user 轮里，每个一段 `<tool_response>`）、生成提示（非思考模式带空的 think 块）。
//! 每次 append 的增量都从 `<|im_end|>` 这个特殊 token 开始，分词边界和整段分词一致。
//!
//! 唯一有意的不同：Qwen3 模板在重渲染历史时会删掉旧 assistant 轮的 think 块
//! （非思考模式下就是那个空的 `<think>\n\n</think>\n\n`）。有状态会话里 KV 保留的是
//! 模型生成时真实看到的 token，不回头改写历史。`kdn-server check-template` 会验证这一点。

use serde_json::Value;
use std::fmt::Write;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

/// Python `json.dumps(x, ensure_ascii=False)` 的输出格式（分隔符 ", " 和 ": "，保持键顺序），
/// 即 HF / llama.cpp 渲染模板时 `tojson` 的结果。
pub fn py_json(v: &Value) -> String {
    let mut s = String::new();
    py_json_into(v, &mut s);
    s
}

fn py_json_into(v: &Value, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_into(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                py_json_into(x, out);
            }
            out.push('}');
        }
        // 字符串：serde_json 不转义非 ASCII，与 ensure_ascii=False 一致
        _ => out.push_str(&serde_json::to_string(v).unwrap()),
    }
}

/// 会话开头：系统提示 + 工具定义
pub fn system_text(system: &str, tools: &[Value]) -> String {
    let mut s = String::new();
    if tools.is_empty() {
        let _ = write!(s, "{IM_START}system\n{system}{IM_END}\n");
        return s;
    }
    let _ = write!(s, "{IM_START}system\n{system}\n\n");
    s.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    for t in tools {
        s.push('\n');
        s.push_str(&py_json(t));
    }
    s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    s
}

/// 一次 append 的增量文本。
/// pending_close：上一轮 assistant 生成停在 `<|im_end|>` 之前（该 token 还没进 KV）。
pub fn delta_text(pending_close: bool, msgs: &[Value], thinking: bool) -> Result<String, String> {
    if msgs.is_empty() {
        return Err("messages 不能为空".into());
    }
    let mut s = String::new();
    if pending_close {
        let _ = write!(s, "{IM_END}\n");
    }
    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).ok_or("消息缺少 role")?;
        let content = content_str(m);
        match role {
            "user" | "system" => {
                let _ = write!(s, "{IM_START}{role}\n{content}{IM_END}\n");
            }
            "tool" => {
                let prev_tool = i > 0 && msgs[i - 1].get("role").and_then(Value::as_str) == Some("tool");
                let next_tool = msgs.get(i + 1).and_then(|n| n.get("role")).and_then(Value::as_str) == Some("tool");
                if !prev_tool {
                    let _ = write!(s, "{IM_START}user");
                }
                let _ = write!(s, "\n<tool_response>\n{content}\n</tool_response>");
                if !next_tool {
                    let _ = write!(s, "{IM_END}\n");
                }
            }
            "assistant" => return Err("assistant 消息由服务端生成，不能 append".into()),
            r => return Err(format!("未知 role: {r}")),
        }
    }
    s.push_str(&gen_prompt(thinking));
    Ok(s)
}

pub fn gen_prompt(thinking: bool) -> String {
    let mut s = format!("{IM_START}assistant\n");
    if !thinking {
        s.push_str(EMPTY_THINK);
    }
    s
}

pub fn content_str(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(v) => v.to_string(),
    }
}

/// assistant 消息按模板渲染出来的文本（不含 `<|im_start|>assistant\n` 和 `<|im_end|>`），
/// 只用于模板自检：模拟"模型生成了这段"。
pub fn assistant_body(content: &str, tool_calls: &[(String, Value)]) -> String {
    let mut s = content.to_string();
    for (i, (name, args)) in tool_calls.iter().enumerate() {
        if i > 0 || !content.is_empty() {
            s.push('\n');
        }
        let _ = write!(s, "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {}}}\n</tool_call>", py_json(args));
    }
    s
}

/// 用 GGUF 里真实的 Jinja 模板渲染（minijinja + Python 字符串方法兼容）
pub fn render_jinja(tmpl: &str, messages: &Value, tools: &Value, add_gen: bool, thinking: bool) -> Result<String, String> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", |v: minijinja::Value| -> Result<String, minijinja::Error> {
        let j: Value = serde_json::to_value(&v).map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
        Ok(py_json(&j))
    });
    env.add_function("raise_exception", |m: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
    });
    env.add_template("chat", tmpl).map_err(|e| e.to_string())?;
    let t = env.get_template("chat").map_err(|e| e.to_string())?;
    let mut ctx = serde_json::json!({
        "messages": messages,
        "add_generation_prompt": add_gen,
        "enable_thinking": thinking,
        "bos_token": "",
        "eos_token": IM_END,
    });
    if tools.as_array().is_some_and(|a| !a.is_empty()) {
        ctx["tools"] = tools.clone();
    }
    t.render(ctx).map_err(|e| format!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_json_matches_python() {
        let v = json!({"a": 1, "b": [1, "x"], "c": {"d": "中文\n"}});
        assert_eq!(py_json(&v), r#"{"a": 1, "b": [1, "x"], "c": {"d": "中文\n"}}"#);
    }

    #[test]
    fn tool_group() {
        let d = delta_text(true, &[json!({"role":"tool","content":"A"}), json!({"role":"tool","content":"B"})], false).unwrap();
        assert_eq!(d, "<|im_end|>\n<|im_start|>user\n<tool_response>\nA\n</tool_response>\n<tool_response>\nB\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
}
