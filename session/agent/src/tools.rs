//! 本地工具：仓库和命令都留在客户端，只有结果（截断后）作为增量发给服务端。
//!
//! 安全：文件工具只能访问工作目录内（规范化后检查前缀，挡住 ../ 和符号链接逃逸）；
//! bash、write_file、edit_file 默认要交互确认（--yes 自动同意）。

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MAX_OUT: usize = 8000;

pub fn definitions() -> Vec<Value> {
    let f = |name: &str, desc: &str, props: Value, req: &[&str]| {
        json!({"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    };
    vec![
        f("read_file", "Read a text file in the repository. Returns its content.", json!({"path": {"type": "string", "description": "Path relative to the repository root"}, "offset": {"type": "integer", "description": "First line to read (1-based, optional)"}, "limit": {"type": "integer", "description": "Max lines to read (optional)"}}), &["path"]),
        f("list_dir", "List files and directories (recursive, depth 2, respects .gitignore).", json!({"path": {"type": "string", "description": "Directory relative to the repository root, default ."}}), &[]),
        f("grep", "Search file contents with a regular expression. Returns file:line: text.", json!({"pattern": {"type": "string"}, "path": {"type": "string", "description": "Directory or file to search, default ."}}), &["pattern"]),
        f("write_file", "Create or overwrite a file with the given content.", json!({"path": {"type": "string"}, "content": {"type": "string"}}), &["path", "content"]),
        f("edit_file", "Replace an exact, unique occurrence of old_string with new_string in a file.", json!({"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}), &["path", "old_string", "new_string"]),
        f("bash", "Run a shell command in the repository root and return stdout+stderr and the exit code.", json!({"command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "Timeout in seconds, default 60"}}), &["command"]),
    ]
}

pub fn needs_confirm(name: &str) -> bool {
    matches!(name, "bash" | "write_file" | "edit_file")
}

/// 预计耗时（秒），作为等待提示发给服务端；None 让服务端按历史预测
pub fn expected_s(name: &str, args: &Value) -> Option<f64> {
    match name {
        "bash" => {
            let c = args["command"].as_str().unwrap_or("");
            let slow = ["make", "cargo ", "npm ", "pytest", "go test", "go build", "cmake", "gcc", "clang", "cc ", "pip "];
            slow.iter().any(|k| c.contains(k)).then_some(20.0)
        }
        _ => Some(0.5),
    }
}

pub struct Tools {
    pub root: PathBuf,
}

fn truncate(s: String) -> String {
    if s.len() <= MAX_OUT {
        return s;
    }
    let cut = |i: usize| (0..=i).rev().find(|&j| s.is_char_boundary(j)).unwrap();
    let head = cut(MAX_OUT * 2 / 3);
    let tail = (s.len() - MAX_OUT / 3..s.len()).find(|&j| s.is_char_boundary(j)).unwrap();
    format!("{}\n...[truncated {} bytes]...\n{}", &s[..head], tail - head, &s[tail..])
}

impl Tools {
    /// 把模型给的路径解析到工作目录内；越界就拒绝
    fn resolve(&self, p: &str) -> Result<PathBuf, String> {
        let p = p.trim();
        let rel = Path::new(if p.is_empty() { "." } else { p });
        let joined = if rel.is_absolute() { rel.to_path_buf() } else { self.root.join(rel) };
        // 先做词法规范化，再对已存在的最长前缀做 canonicalize（挡符号链接）
        let mut norm = PathBuf::new();
        for c in joined.components() {
            match c {
                Component::ParentDir => {
                    norm.pop();
                }
                Component::CurDir => {}
                c => norm.push(c),
            }
        }
        let mut existing = norm.clone();
        let mut rest = vec![];
        while !existing.exists() {
            match (existing.file_name().map(|s| s.to_owned()), existing.parent()) {
                (Some(n), Some(par)) => {
                    rest.push(n);
                    existing = par.to_path_buf();
                }
                _ => break,
            }
        }
        let mut real = existing.canonicalize().map_err(|e| format!("{p}: {e}"))?;
        for n in rest.into_iter().rev() {
            real.push(n);
        }
        if !real.starts_with(&self.root) {
            return Err(format!("拒绝访问工作目录以外的路径：{p}"));
        }
        Ok(real)
    }

    fn rel(&self, p: &Path) -> String {
        p.strip_prefix(&self.root).map(|r| r.display().to_string()).unwrap_or_else(|_| p.display().to_string())
    }

    /// 执行一次工具调用，返回给模型看的文本
    pub fn run(&self, name: &str, args: &Value) -> String {
        let r = match name {
            "read_file" => self.read_file(args),
            "list_dir" => self.list_dir(args),
            "grep" => self.grep(args),
            "write_file" => self.write_file(args),
            "edit_file" => self.edit_file(args),
            "bash" => self.bash(args),
            _ => Err(format!("unknown tool: {name}")),
        };
        truncate(r.unwrap_or_else(|e| format!("Error: {e}")))
    }

    fn arg<'a>(args: &'a Value, k: &str) -> Result<&'a str, String> {
        args.get(k).and_then(Value::as_str).ok_or_else(|| format!("missing argument '{k}'"))
    }

    fn read_file(&self, a: &Value) -> Result<String, String> {
        let p = self.resolve(Self::arg(a, "path")?)?;
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", self.rel(&p)))?;
        let off = a.get("offset").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
        let lim = a.get("limit").and_then(Value::as_u64).unwrap_or(400) as usize;
        let lines: Vec<&str> = text.lines().collect();
        let end = (off - 1 + lim).min(lines.len());
        let body = lines.get(off - 1..end).unwrap_or(&[]).join("\n");
        let more = if end < lines.len() { format!("\n[... {} more lines; use offset={}]", lines.len() - end, end + 1) } else { String::new() };
        Ok(format!("{body}{more}"))
    }

    fn list_dir(&self, a: &Value) -> Result<String, String> {
        let p = self.resolve(a.get("path").and_then(Value::as_str).unwrap_or("."))?;
        let mut out = vec![];
        for e in ignore::WalkBuilder::new(&p).max_depth(Some(2)).hidden(false).filter_entry(|e| e.file_name() != ".git").build().flatten().skip(1).take(300) {
            let mut s = self.rel(e.path());
            if e.file_type().is_some_and(|t| t.is_dir()) {
                s.push('/');
            }
            out.push(s);
        }
        out.sort();
        Ok(if out.is_empty() { "(empty)".into() } else { out.join("\n") })
    }

    fn grep(&self, a: &Value) -> Result<String, String> {
        let re = regex::Regex::new(Self::arg(a, "pattern")?).map_err(|e| e.to_string())?;
        let p = self.resolve(a.get("path").and_then(Value::as_str).unwrap_or("."))?;
        let mut out = vec![];
        'walk: for e in ignore::WalkBuilder::new(&p).hidden(false).filter_entry(|e| e.file_name() != ".git").build().flatten() {
            if !e.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    out.push(format!("{}:{}: {}", self.rel(e.path()), i + 1, line.trim_end()));
                    if out.len() >= 100 {
                        out.push("[... more matches omitted]".into());
                        break 'walk;
                    }
                }
            }
        }
        Ok(if out.is_empty() { "(no matches)".into() } else { out.join("\n") })
    }

    fn write_file(&self, a: &Value) -> Result<String, String> {
        let p = self.resolve(Self::arg(a, "path")?)?;
        let c = Self::arg(a, "content")?;
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, c).map_err(|e| e.to_string())?;
        Ok(format!("Wrote {} bytes to {}", c.len(), self.rel(&p)))
    }

    fn edit_file(&self, a: &Value) -> Result<String, String> {
        let p = self.resolve(Self::arg(a, "path")?)?;
        let old = Self::arg(a, "old_string")?;
        let new = Self::arg(a, "new_string")?;
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", self.rel(&p)))?;
        match text.matches(old).count() {
            0 => Err(format!("old_string not found in {}", self.rel(&p))),
            1 => {
                std::fs::write(&p, text.replacen(old, new, 1)).map_err(|e| e.to_string())?;
                Ok(format!("Edited {}", self.rel(&p)))
            }
            n => Err(format!("old_string occurs {n} times in {}; make it unique", self.rel(&p))),
        }
    }

    fn bash(&self, a: &Value) -> Result<String, String> {
        let cmd = Self::arg(a, "command")?;
        let timeout = Duration::from_secs(a.get("timeout_s").and_then(Value::as_u64).unwrap_or(60));
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(format!("{cmd} 2>&1"))
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut buf = vec![];
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let t0 = Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
                break Some(s);
            }
            if t0.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let out = String::from_utf8_lossy(&reader.join().unwrap_or_default()).into_owned();
        Ok(match status {
            Some(s) => format!("{out}\n[exit code: {}]", s.code().map_or("signal".into(), |c| c.to_string())),
            None => format!("{out}\n[killed after {}s timeout]", timeout.as_secs()),
        })
    }
}

/// 交互确认；返回 (是否同意, 以后都同意)
pub fn confirm(prompt: &str) -> (bool, bool) {
    print!("{prompt} [y/N/a=always] ");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    if std::io::stdin().read_line(&mut s).is_err() {
        return (false, false);
    }
    match s.trim().to_lowercase().as_str() {
        "y" | "yes" => (true, false),
        "a" | "always" => (true, true),
        _ => (false, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox() {
        let d = std::env::temp_dir().join(format!("kdn-tools-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let t = Tools { root: d.canonicalize().unwrap() };
        assert!(t.resolve("a/b.txt").is_ok());
        assert!(t.resolve("../x").is_err());
        assert!(t.resolve("/etc/passwd").is_err());
        assert!(t.resolve("a/../../x").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
