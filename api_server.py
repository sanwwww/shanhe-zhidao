# -*- coding: utf-8 -*-
"""
山河智导 · FastAPI 服务层（SSE 流式 + 会话记忆 + 安全防护）
- lifespan：启动时连接全部 MCP Server 并构建 Agent，关闭时清理（教程同款生命周期管理）
- SSE 事件协议与 Ops Agent 一致：status / delta / tool_start / tool_result / done / error
- 安全：每 IP 滑动窗口限流 30 次/分、消息 ≤500 字、request_id 全链路追踪
- 可靠性：90s 墙钟熔断 + ReAct 步数上限 + 断连即停 + 异常不回传堆栈
  + MCP 会话断线自愈（异常触发重建 + 60s 健康巡检兜底）
跑法：python api_server.py  （默认 http://127.0.0.1:8001）
"""
import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent_core import ROOT, AgentHolder, is_session_broken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shanhe")

RATE_LIMIT = 30           # 每 IP 每分钟（原 10：演示场景下连问几题即 429，过严）
MAX_MSG_LEN = 500
TOTAL_TIMEOUT = 90        # 单次请求墙钟上限（秒）：LLM 或 MCP 挂起时主动熔断，避免 SSE 无限挂起
MAX_STEPS = 12            # ReAct 循环步数上限（有界自主，防跑飞）
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
    # 应用启动时初始化 MCP 会话与 Agent；关闭时清理资源
    holder = AgentHolder()
    await holder.start()          # 内含常驻 supervisor task（会话的建立/关闭都在它里面）
    app.state.holder = holder
    print(f"[山河智导] Agent 就绪，接入工具：{[t.name for t in holder.tools]}"
          + (f"　未连上：{holder.failures}" if holder.failures else ""))
    try:
        yield
    finally:
        await holder.aclose()


app = FastAPI(title="山河智导 · 文旅导览 Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class AnswerStreamFilter:
    """从 ReAct 事件流中筛出真正属于答案的文本。

    ReAct 会多次调用模型：中间轮次的 content 只是“前言/思考”，实测会输出
    "I'll look this up in the knowledge base." 这类英文句子；只有最后一轮才是答案。
    策略：按 run_id 把每次 LLM 调用切成独立段，段内触发过工具即判为前言并丢弃。

    不用“文本长度阈值”判断，是因为实测前言可达 60+ 字符，阈值法无法可靠区分；
    也不用“首次工具前的内容都丢弃”，因为多步调用时每一轮前都可能有前言。
    """

    def __init__(self) -> None:
        self._seg: dict | None = None
        self._pending: list[str] = []

    def on_model_chunk(self, run_id, text: str) -> None:
        if not text:
            return
        if self._seg is None or (run_id and self._seg["rid"] != run_id):
            self._close()                      # 上一段结束，结算
            self._seg = {"rid": run_id, "texts": [], "had_tool": False}
        self._seg["texts"].append(text)

    def on_tool_start(self) -> None:
        """工具在本段 LLM 之后触发 → 本段 content 是前言，不是答案"""
        if self._seg is not None:
            self._seg["had_tool"] = True

    def drain(self) -> list[str]:
        """取出并清空自上次 drain 以来已确认属于答案的文本"""
        out, self._pending = self._pending, []
        return out

    def finish(self) -> list[str]:
        self._close()                          # 最后一段即正式回答
        return self.drain()

    def _close(self) -> None:
        if self._seg is None:
            return
        if not self._seg["had_tool"]:
            self._pending.extend(self._seg["texts"])
        self._seg = None


def _to_text(out) -> str:
    """工具返回值 → 纯文本。

    LangChain v1 下 MCP 工具的输出是 content-blocks 列表（形如 [{'type':'text','text':...}]），
    直接 str() 会把 Python repr 推给前端，必须按块取出真正的文本。
    这里不依赖任何特定版本的内部结构，按 str / list / dict / 对象 四种形态兜底。
    """
    if isinstance(out, str):
        return out
    content = getattr(out, "content", out)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(getattr(block, "text", block)))
        return "".join(p for p in parts if p)
    return "" if content is None else str(content)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
    holder = getattr(request.app.state, "holder", None)
    if holder is None or not holder.ready:
        # 会话重建期间（几秒）没有可用 Agent：明确让用户稍后重试，
        # 而不是把一个指向已关闭会话的旧 Agent 拿去撞墙
        return JSONResponse({"detail": "正在重置工具连接，请 5 秒后重试"}, status_code=503)
    msg = req.message.strip()
    if not msg or len(msg) > MAX_MSG_LEN:
        return JSONResponse({"detail": f"消息不能为空且不超过{MAX_MSG_LEN}字"}, status_code=400)

    thread_id = req.thread_id or uuid.uuid4().hex[:12]
    request_id = uuid.uuid4().hex[:12]
    started = time.time()

    async def gen():
        steps = 0
        filt = AnswerStreamFilter()

        try:
            yield _sse("status", {"stage": "connecting", "request_id": request_id})
            # 每次请求现取 agent：会话重建后新请求自动用上新的一套
            stream = holder.agent.astream_events(
                {"messages": [HumanMessage(content=msg)]},
                {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS * 2 + 1},
                version="v2",
            )
            # 墙钟上限：LLM 或 MCP 子进程挂起时主动熔断，而不是让 SSE 流无限挂着转圈
            async with asyncio.timeout(TOTAL_TIMEOUT):
                async for ev in stream:
                    if await request.is_disconnected():
                        return                      # 客户端已断开，停止生成，别浪费 token
                    kind = ev["event"]
                    if kind == "on_chat_model_stream":
                        chunk = ev.get("data", {}).get("chunk")
                        text = getattr(chunk, "content", "") if chunk else ""
                        if text:
                            filt.on_model_chunk(ev.get("run_id"), text)
                            for t in filt.drain():
                                yield _sse("delta", {"text": t})
                    elif kind == "on_tool_start":
                        filt.on_tool_start()
                        steps += 1
                        yield _sse("tool_start", {"tool": ev["name"], "args": ev.get("data", {}).get("input")})
                    elif kind == "on_tool_end":
                        out = ev.get("data", {}).get("output")
                        yield _sse("tool_result", {"tool": ev["name"], "result": _to_text(out)[:800]})
            for t in filt.finish():
                yield _sse("delta", {"text": t})
            yield _sse("done", {
                "thread_id": thread_id, "steps": steps,
                "elapsed": round(time.time() - started, 1),
                "request_id": request_id,
            })
        except asyncio.TimeoutError:
            log.warning("[%s] timeout after %ss", request_id, TOTAL_TIMEOUT)
            yield _sse("error", {"answer": f"响应超过 {TOTAL_TIMEOUT} 秒已中断，请简化问题后重试。",
                                 "request_id": request_id})
        except Exception as e:
            if is_session_broken(e):
                # MCP 会话已断：后台重建整套会话，本次如实告知用户重试一次即可
                log.warning("[%s] MCP 会话断开（%s），已触发后台重建",
                            request_id, type(e).__name__)
                app.state.holder.trigger_rebuild()
                yield _sse("error", {"answer": "工具连接已重置，请重试一次。",
                                     "request_id": request_id})
            else:
                log.exception("[%s] chat failed: %s", request_id, e)   # 详情只进服务端日志
                yield _sse("error", {"answer": "服务暂时不可用，请稍后重试。",
                                     "request_id": request_id})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"})


@app.get("/health")
async def health(request: Request):
    """健康检查。/health 是人（和监控）用来判断"它还活着吗"的入口，
    所以要把"会话有没有断过、有没有 Server 没连上"一起暴露出来——
    否则只能看到 status: ok 却不知道工具其实已经失效。"""
    holder = getattr(request.app.state, "holder", None)
    if holder is None:
        return JSONResponse({"status": "starting", "service": "shanhe-zhidao"}, status_code=503)
    if not holder.ready:
        # 正在重建会话：明确区别于"健康"，否则监控会以为一切正常
        return JSONResponse({"status": "recovering", "service": "shanhe-zhidao",
                             "session_rebuilds": holder.rebuilds}, status_code=503)
    return {
        "status": "ok",
        "service": "shanhe-zhidao",
        "tools": [t.name for t in holder.tools],
        "unavailable_servers": holder.failures,   # 非空即说明某个 Server 的工具当前不可用
        "session_rebuilds": holder.rebuilds,      # 会话重建次数：持续增长 = 子进程反复挂
    }


# 静态前端（放最后，避免覆盖 API 路由）
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
