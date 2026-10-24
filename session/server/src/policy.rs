//! 等待信号 -> 预测空档 -> 驱逐键。逻辑照搬 src/common.h。

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sig {
    Local,
    Mcp,
    Agent,
    Monitor,
    Wakeup,
    Bash,
    Web,
    Other,
    Human,
    Ask,
    End,
}
pub const N_SIG: usize = 11;

/// 各信号的默认预测空档（秒）
const DEFAULT_GAP: [f64; N_SIG] = [0.5, 1.0, 1.0, 1.0, 1.0, 3.0, 20.0, 5.0, 60.0, 120.0, 3600.0];
/// 会话内有历史时，是否用上一次同类信号的实际空档
const USE_HISTORY: [bool; N_SIG] = [false, false, false, false, false, true, true, true, true, true, false];

impl Sig {
    /// 把 hint 里的 waiting / tool 映射成信号
    pub fn from_hint(waiting: &str, tool: Option<&str>) -> Option<Sig> {
        Some(match waiting {
            "human" => Sig::Human,
            "ask" => Sig::Ask,
            "done" => Sig::End,
            "tool" => match tool.unwrap_or("") {
                "bash" | "shell" => Sig::Bash,
                "read_file" | "list_dir" | "grep" | "search" | "write_file" | "edit_file" | "local" => Sig::Local,
                "web" | "web_fetch" | "web_search" => Sig::Web,
                "mcp" => Sig::Mcp,
                "agent" | "subagent" => Sig::Agent,
                _ => Sig::Other,
            },
            _ => return None,
        })
    }
    pub fn is_human(self) -> bool {
        matches!(self, Sig::Human | Sig::Ask)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Predictor {
    /// 上一次同类信号的实际空档，<0 表示没有
    pub last: [f64; N_SIG],
}

impl Default for Predictor {
    fn default() -> Self {
        Predictor { last: [-1.0; N_SIG] }
    }
}

impl Predictor {
    pub fn gap(&self, sig: Sig) -> f64 {
        let i = sig as usize;
        if USE_HISTORY[i] && self.last[i] >= 0.0 {
            self.last[i]
        } else {
            DEFAULT_GAP[i]
        }
    }
    pub fn observe(&mut self, sig: Sig, gap: f64) {
        if USE_HISTORY[sig as usize] {
            self.last[sig as usize] = gap;
        }
    }
}

/// 驱逐优先级：越大越先挪走。end = 等待开始，gap = 预测空档。
/// 超时未到：剩余时间；已超时：已等待 w 秒的重尾等待，期望剩余约等于 w。
pub fn evict_key(now: f64, end: f64, gap: f64) -> f64 {
    let remaining = end + gap - now;
    if remaining >= 0.0 {
        remaining
    } else {
        now - end
    }
}
