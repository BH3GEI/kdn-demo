//! HTTP JSON API（axum）。所有请求转成命令丢给引擎线程；append 用 SSE 流式返回。

use crate::engine::{ApiErr, AppendReq, Cmd, CreateReq, Event, HintReq};
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::mpsc::Sender;
use tokio::sync::oneshot;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

#[derive(Clone)]
pub struct AppState {
    pub tx: Sender<Cmd>,
}

impl IntoResponse for ApiErr {
    fn into_response(self) -> Response {
        let code = StatusCode::from_u16(self.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (code, Json(json!({"error": self.1}))).into_response()
    }
}

type R = Result<Json<Value>, ApiErr>;

/// 发命令给引擎并等回复
async fn call(st: &AppState, f: impl FnOnce(oneshot::Sender<Result<Value, ApiErr>>) -> Cmd) -> R {
    let (tx, rx) = oneshot::channel();
    st.tx.send(f(tx)).map_err(|_| ApiErr(503, "引擎已停止".into()))?;
    rx.await.map_err(|_| ApiErr(500, "引擎没有回复".into()))?.map(Json)
}

fn parse<T: DeserializeOwned>(body: &[u8]) -> Result<T, ApiErr> {
    let body = if body.is_empty() { b"{}".as_slice() } else { body };
    serde_json::from_slice(body).map_err(|e| ApiErr(400, format!("请求 JSON 不对：{e}")))
}

pub fn router(st: AppState) -> Router {
    Router::new()
        .route("/v1/sessions", post(create).get(list))
        .route("/v1/sessions/{id}", get(get_one).delete(delete))
        .route("/v1/sessions/{id}/append", post(append))
        .route("/v1/sessions/{id}/hint", post(hint))
        .route("/v1/sessions/{id}/fork", post(fork))
        .route("/v1/sessions/{id}/rollback", post(rollback))
        .route("/v1/admin/sessions/{id}/demote", post(demote))
        .route("/v1/admin/stats", get(stats))
        .with_state(st)
}

async fn create(State(st): State<AppState>, body: Bytes) -> R {
    let req: CreateReq = parse(&body)?;
    call(&st, |r| Cmd::Create(req, r)).await
}

async fn list(State(st): State<AppState>) -> R {
    call(&st, Cmd::List).await
}

#[derive(serde::Deserialize)]
struct GetQ {
    #[serde(default)]
    messages: bool,
}

async fn get_one(State(st): State<AppState>, Path(id): Path<String>, Query(q): Query<GetQ>) -> R {
    call(&st, |r| Cmd::Get(id, q.messages, r)).await
}

async fn delete(State(st): State<AppState>, Path(id): Path<String>) -> R {
    call(&st, |r| Cmd::Delete(id, r)).await
}

async fn hint(State(st): State<AppState>, Path(id): Path<String>, body: Bytes) -> R {
    let h: HintReq = parse(&body)?;
    call(&st, |r| Cmd::Hint(id, h, r)).await
}

async fn fork(State(st): State<AppState>, Path(id): Path<String>, body: Bytes) -> R {
    let v: Value = parse(&body)?;
    let turn = v.get("turn").and_then(Value::as_u64).map(|n| n as usize);
    call(&st, |r| Cmd::Fork(id, turn, r)).await
}

async fn rollback(State(st): State<AppState>, Path(id): Path<String>, body: Bytes) -> R {
    let v: Value = parse(&body)?;
    let turn = v.get("turn").and_then(Value::as_u64).ok_or_else(|| ApiErr(400, "需要 turn".into()))? as usize;
    call(&st, |r| Cmd::Rollback(id, turn, r)).await
}

async fn demote(State(st): State<AppState>, Path(id): Path<String>, body: Bytes) -> R {
    let v: Value = parse(&body)?;
    let tier = v.get("tier").and_then(Value::as_str).unwrap_or("ssd").to_string();
    call(&st, |r| Cmd::Demote(id, tier, r)).await
}

async fn stats(State(st): State<AppState>) -> R {
    call(&st, Cmd::Stats).await
}

/// append：默认 SSE（event: token / done / error）；"stream": false 时一次性返回 JSON
async fn append(State(st): State<AppState>, Path(id): Path<String>, body: Bytes) -> Response {
    let v: Value = match parse(&body) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let mut req: AppendReq = match serde_json::from_value(v) {
        Ok(r) => r,
        Err(e) => return ApiErr(400, format!("请求 JSON 不对：{e}")).into_response(),
    };
    req.request_bytes = body.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    if st.tx.send(Cmd::Append(id, req, tx)).is_err() {
        return ApiErr(503, "引擎已停止".into()).into_response();
    }
    if stream {
        let s = UnboundedReceiverStream::new(rx).map(|ev| {
            Ok::<_, Infallible>(match ev {
                Event::Token(t) => SseEvent::default().event("token").data(json!({"t": t}).to_string()),
                Event::Done(v) => SseEvent::default().event("done").data(v.to_string()),
                Event::Error(e) => SseEvent::default().event("error").data(json!({"error": e}).to_string()),
            })
        });
        return Sse::new(s).into_response();
    }
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token(_) => {}
            Event::Done(v) => return Json(v).into_response(),
            Event::Error(e) => return ApiErr(400, e).into_response(),
        }
    }
    ApiErr(500, "引擎没有回复".into()).into_response()
}
