# -*- coding: utf-8 -*-
"""
山河智导 · FastAPI 服务层（SSE 流式 + 会话记忆 + 安全防护）
- lifespan：启动时连接全部 MCP Server 并构建 Agent，关闭时清理（教程同款生命周期管理）
- SSE 事件协议与 Ops Agent 一致：status / delta / tool_start / tool_result / done / error
- 安全：每 IP 滑动窗口限流 10 次/分、消息 ≤500 字、request_id 全链路追踪
跑法：python api_server.py  （默认 http://127.0.0.1:8001）
"""
import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent_core import ROOT, create_app_agent

RATE_LIMIT = 10          # 每 IP 每分钟
MAX_MSG_LEN = 500
_hits: dict[str, deque] = defaultdict(deque)


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时初始化 MCP 连接和 Agent；关闭时清理资源
    async with create_app_agent() as (agent, tools):
        app.state.agent = agent
        app.state.tool_names = [t.name for t in tools]
        print(f"[山河智导] Agent 就绪，接入工具：{app.state.tool_names}")
        yield


app = FastAPI(title="山河智导 · 文旅导览 Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
    msg = req.message.strip()
    if not msg or len(msg) > MAX_MSG_LEN:
        return JSONResponse({"detail": f"消息不能为空且不超过{MAX_MSG_LEN}字"}, status_code=400)

    thread_id = req.thread_id or uuid.uuid4().hex[:12]
    request_id = uuid.uuid4().hex[:12]
    started = time.time()

    async def gen():
        steps = 0
        answer_parts: list[str] = []
        try:
            yield _sse("status", {"stage": "connecting", "request_id": request_id})
            stream = app.state.agent.astream_events(
                {"messages": [HumanMessage(content=msg)]},
                {"configurable": {"thread_id": thread_id}},
                version="v2",
            )
            async for ev in stream:
                kind = ev["event"]
                if kind == "on_tool_start":
                    steps += 1
                    yield _sse("tool_start", {"tool": ev["name"], "args": ev.get("data", {}).get("input")})
                elif kind == "on_tool_end":
                    out = ev.get("data", {}).get("output")
                    text = getattr(out, "content", out)
                    yield _sse("tool_result", {"tool": ev["name"], "result": str(text)[:800]})
                elif kind == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")
                    text = getattr(chunk, "content", "") if chunk else ""
                    if text:
                        answer_parts.append(text)
                        yield _sse("delta", {"text": text})
            yield _sse("done", {
                "thread_id": thread_id, "steps": steps,
                "elapsed": round(time.time() - started, 1),
                "request_id": request_id,
            })
        except Exception as e:
            yield _sse("error", {"answer": f"服务异常：{e}", "request_id": request_id})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"})


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "service": "shanhe-zhidao",
            "tools": getattr(request.app.state, "tool_names", [])}


# 静态前端（放最后，避免覆盖 API 路由）
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
