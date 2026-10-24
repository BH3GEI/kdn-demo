//! 有状态会话服务的 HTTP 客户端（同步）。append 走 SSE，逐 token 回调。

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};

pub struct Client {
    base: String,
    http: reqwest::blocking::Client,
    /// 本会话累计发出的请求体字节（含 hint 等小请求）
    pub sent_bytes: usize,
}

impl Client {
    pub fn new(base: &str) -> Client {
        let http = reqwest::blocking::Client::builder().timeout(None).build().unwrap();
        Client { base: base.trim_end_matches('/').to_string(), http, sent_bytes: 0 }
    }

    fn send(&mut self, method: reqwest::Method, path: &str, body: Option<&Value>) -> Result<reqwest::blocking::Response> {
        let mut rb = self.http.request(method, format!("{}{path}", self.base));
        if let Some(b) = body {
            let s = serde_json::to_vec(b)?;
            self.sent_bytes += s.len();
            rb = rb.header("content-type", "application/json").body(s);
        }
        let r = rb.send().map_err(|e| anyhow!("连不上服务 {}：{e}", self.base))?;
        if !r.status().is_success() {
            let code = r.status();
            let t = r.text().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&t).ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or(t);
            bail!("{code}: {msg}");
        }
        Ok(r)
    }

    pub fn get(&mut self, path: &str) -> Result<Value> {
        Ok(self.send(reqwest::Method::GET, path, None)?.json()?)
    }
    pub fn post(&mut self, path: &str, body: &Value) -> Result<Value> {
        Ok(self.send(reqwest::Method::POST, path, Some(body))?.json()?)
    }
    pub fn delete(&mut self, path: &str) -> Result<Value> {
        Ok(self.send(reqwest::Method::DELETE, path, None)?.json()?)
    }

    pub fn create(&mut self, system: &str, tools: &[Value], thinking: Option<bool>, label: &str) -> Result<Value> {
        let mut b = json!({"system": system, "tools": tools, "label": label});
        if let Some(t) = thinking {
            b["thinking"] = json!(t);
        }
        self.post("/v1/sessions", &b)
    }

    pub fn hint(&mut self, id: &str, waiting: &str, tool: Option<&str>, expected_s: Option<f64>) -> Result<Value> {
        let mut b = json!({"waiting": waiting});
        if let Some(t) = tool {
            b["tool"] = json!(t);
        }
        if let Some(e) = expected_s {
            b["expected_s"] = json!(e);
        }
        self.post(&format!("/v1/sessions/{id}/hint"), &b)
    }

    /// 追加增量并生成；on_token 收到流式文本。返回 done 事件的内容和本次请求体字节数。
    pub fn append(&mut self, id: &str, body: &Value, mut on_token: impl FnMut(&str)) -> Result<(Value, usize)> {
        let before = self.sent_bytes;
        let r = self.send(reqwest::Method::POST, &format!("/v1/sessions/{id}/append"), Some(body))?;
        let sent = self.sent_bytes - before;
        let mut event = String::new();
        let mut data = String::new();
        for line in BufReader::new(r).lines() {
            let line = line?;
            if let Some(e) = line.strip_prefix("event:") {
                event = e.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                data.push_str(d.strip_prefix(' ').unwrap_or(d));
            } else if line.is_empty() && !data.is_empty() {
                let v: Value = serde_json::from_str(&data)?;
                data.clear();
                match event.as_str() {
                    "token" => on_token(v["t"].as_str().unwrap_or("")),
                    "done" => return Ok((v, sent)),
                    "error" => bail!("服务端错误：{}", v["error"].as_str().unwrap_or("?")),
                    _ => {}
                }
            }
        }
        bail!("流在 done 之前断了")
    }
}
