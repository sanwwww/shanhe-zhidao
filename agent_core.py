# -*- coding: utf-8 -*-
"""
山河智导 · Agent 组装核心（被 client.py / api_server.py / evaluate.py 共用）
- MultiServerMCPClient：按 servers_config.json 拉起全部 MCP Server，自动发现工具
- AgentHolder：把「一组长期 MCP 会话 + Agent」打包成可原子替换的单元，断线可重建
- create_react_agent：LangGraph 预建 ReAct 循环（= Ops Agent 里手写的那个循环的框架封装版）
- checkpointer：多轮记忆，thread_id 隔离会话
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

log = logging.getLogger("shanhe")

SYSTEM_PROMPT = """你是"山河智导"，一位专业的陕西文旅导览 Agent，由一位有博物院讲解经验的开发者打造。

工作规则：
1. 先判断用户意图类型：路线交通 / 天气穿搭 / 景点文物知识 / 讲解词 / 行程规划（复合型）。
2. 景点、文物、攻略、行程类问题，必须先用 search_knowledge 检索知识库再回答，禁止凭模型记忆编造年代、数字、票价。
3. 路线问题用 plan_route，场所位置用 search_place，天气用 get_weather，穿搭建议配合 get_clothing_advice。
4. 用户要求"讲一讲/组织讲解词"时用 generate_guide，并主动询问观众类型（儿童/学生/亲子/老年/外宾/历史爱好者）。
5. 复合型行程规划（如"一天怎么玩"）应组合调用：知识库（景点背景）+ 天气 + 路线，给出结构化方案。
6. 结论结构化：先给直接答案，再给依据（注明来自知识库/实时查询），最后给一个实用的下一步建议。
7. 工具返回"未配置/查询失败"时，诚实告知并给出替代方案，不要假装查到了。
8. 全部工具都是只读查询，你不具备订票、预约的能力，涉及预约要引导用户去官方公众号。
9. 任何具体数值（票价、开放时间与钟点、里程、耗时、电话、优惠政策）只能来自工具返回结果。
   工具结果里没有的，一律回答"知识库未收录，请以官方渠道当日公告为准"，不要给出你印象中的大概率数字——
   先说"我不能凭记忆编造"、随后又报一个具体数字，是自相矛盾的，这种情况宁可少说。
10. 回答天气问题必须引用 get_weather 返回的实测气温（℃）与降水概率，不能只给"偏冷/舒适"这类定性结论。"""


def load_servers_config() -> dict:
    """读 servers_config.json 并把相对路径解析成绝对路径（工作目录无关，更稳）"""
    cfg = json.loads((ROOT / "servers_config.json").read_text(encoding="utf-8"))
    # MCP stdio 默认只转发白名单环境变量（PATH 等），业务密钥必须显式注入子进程；
    # 显式传 env 时会整体替换而非合并，所以先复制完整环境再注入。
    # 注意：这里对所有 server 统一注入，而不是"有 AMAP_KEY 才注入"——
    # 之前只在 AMAP_KEY 存在时才设置 env，导致只配了 QWEATHER_KEY 的用户
    # 天气服务依然拿不到密钥，是个静默失效的坑。
    merged = dict(os.environ)
    merged.setdefault("FASTMCP_LOG_LEVEL", "WARNING")   # 子进程 INFO 日志会混进评测输出，压到 WARNING
    for name, s in cfg.items():
        if s.get("transport") == "stdio":
            s["args"] = [str((ROOT / a).resolve()) if a.startswith("servers/") else a
                         for a in s["args"]]
            # 必须用当前解释器拉起 Server（配置里的 "python" 可能指向无依赖的系统 Python）
            s["command"] = sys.executable
            s["env"] = merged
    return cfg


def build_model() -> ChatOpenAI:
    """DeepSeek（OpenAI 兼容协议）。换模型只改这两行——模型层可替换。"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.3,
        streaming=True,
    )


# 管理/写操作类工具：保留在 MCP Server 供运维单独调用，但不进入 Agent 可见工具列表。
# 权限靠代码边界，不靠 prompt 约束模型“别调用”——prompt 是建议，列表剔除才是保证。
HIDDEN_FROM_AGENT = {"rebuild_knowledge"}

# 会话断开时底层抛出的异常族。
# 实测（_probe_kill.py：taskkill 掉 stdio 子进程后再调工具）：立即抛 anyio.ClosedResourceError，
# 耗时 0.0s，且连续调用稳定复现——即"快速失败"，不会挂起。这让重连策略是可行的。
SESSION_BROKEN_EXC = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    ConnectionError,
    BrokenPipeError,
    EOFError,
)


def _walk_exceptions(exc, depth: int = 0):
    """展开 ExceptionGroup 与 cause/context 链，找到真正的根因。

    anyio 的 task group 会把底层异常包进 ExceptionGroup，langgraph 还可能再包一层，
    所以不能只看最外层类型——那样会漏判，表现为"会话明明断了却不重建"。
    """
    if exc is None or depth > 6:
        return
    yield exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _walk_exceptions(sub, depth + 1)
    yield from _walk_exceptions(exc.__cause__, depth + 1)
    yield from _walk_exceptions(exc.__context__, depth + 1)


def is_session_broken(exc: BaseException) -> bool:
    """异常是否属于"MCP 会话已断"（需要重建整套会话）。

    只认链路级异常：工具自身的业务错误会被 MCP 包成 ToolMessage(status="error")
    交给模型自纠，不会冒到这里；真冒到这里的基本都是连接/进程级故障。
    """
    return any(isinstance(e, SESSION_BROKEN_EXC) for e in _walk_exceptions(exc))


class AgentHolder:
    """把「一组长期 MCP 会话 + 由此组装的 Agent」打包成一个可原子替换的单元。

    **为什么用长期会话**：`client.get_tools()` 的 docstring 明写
    "A new session will be created for each tool call"——每次工具调用都要重开
    stdio 子进程、重新 import 整个依赖栈。实测（bench_concurrency.py）：

        get_tools()      单次工具调用 3.670s
        复用长期会话      单次工具调用 0.209s    ← 快 17.6 倍
        同题端到端对比    均值 6.50s → 4.49s，P50 6.61s → 3.76s（工具调用次数不变）

    **代价与应对**：长期会话一断就全断（实测杀掉子进程后每次调用都抛
    anyio.ClosedResourceError）。所以这里把整组会话和 Agent 做成可原子替换的：
    调用方发现链路异常就 trigger_rebuild()，下一个请求即恢复，不必重启服务。

    **checkpointer 跨重建复用**：它是会话记忆的载体，重建时若新建会丢光历史。
    """

    def __init__(self, health_interval: float = 60.0) -> None:
        self._checkpointer = MemorySaver()
        self._stack: AsyncExitStack | None = None
        self._agent = None
        self._tools: list = []
        self._failures: list[str] = []
        self._sessions: dict = {}
        self._cmds: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._builds = 0
        self._recoveries = 0
        self._health_interval = health_interval
        self._ping_failures: dict[str, int] = {}

    # ── 只读视图（供 /health 与日志观测） ─────────────────────
    @property
    def ready(self) -> bool:
        """是否有可用的 Agent。重建期间为 False——此时请求应快速失败，
        而不是拿到一个指向已关闭会话的旧 Agent。"""
        return self._agent is not None

    @property
    def agent(self):
        if self._agent is None:
            raise RuntimeError("AgentHolder 尚未 open()")
        return self._agent

    @property
    def tools(self) -> list:
        return list(self._tools)

    @property
    def failures(self) -> list[str]:
        """上一次重建时连接失败的 Server 名（空列表 = 全部正常）"""
        return list(self._failures)

    @property
    def builds(self) -> int:
        """成功建立会话的总次数。首次成立即为 1，不是 0。"""
        return self._builds

    @property
    def recoveries(self) -> int:
        """**自愈**重建次数：只有"在已有可用 Agent 的状态下重新建立"才 +1。

        为什么要和 builds 分开：验收断线自愈时，若只有一个从 1 起步的总数，
        看到 1 无法判断是"正常启动"还是"已经挂过一次并自愈"。分开口径后，
        健康实例恒为 0，任何一次增长都等价于"确实自愈了一次"——这个指标才可自证。
        """
        return self._recoveries

    def _note_build(self, was_ready: bool) -> None:
        """记一次成功建立。`was_ready` = 建立之前是否已有可用 Agent。"""
        self._builds += 1
        if was_ready:
            self._recoveries += 1

    async def start(self) -> None:
        """启动常驻会话管理 task，并等待首轮会话就绪。"""
        self._task = asyncio.create_task(self._supervise(), name="mcp-session-supervisor")
        await self._request("rebuild")

    async def rebuild(self) -> None:
        """请求重建整套会话（等待完成）。

        **注意**：真正的建立/关闭全部在常驻 supervisor task 里执行。
        绝不能在这里直接 enter/exit MCP 会话——原因见 `_supervise` 的说明。
        """
        await self._request("rebuild")

    async def _request(self, cmd: str) -> None:
        """向 supervisor 投递指令并等待完成（带超时，避免调用方被永久卡住）。"""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._cmds.put((cmd, fut))
        await asyncio.wait_for(fut, timeout=120)   # 4 个 Server 冷启动约 3–8s，120s 是保险丝

    async def _supervise(self) -> None:
        """常驻 task：MCP 会话的建立与关闭**只在这里**发生。

        为什么必须是常驻 task（这条是踩坑换来的）：
        mcp 的 stdio 会话底层是 anyio 的 task group，anyio 要求 cancel scope
        必须在**创建它的同一个 task** 内退出。第一版把重建放在
        `asyncio.create_task` 里做，退出旧会话时 anyio 取消了一个不属于当前 task
        的 scope——而那个 scope 恰好是 uvicorn lifespan 所在的外层 scope，
        于是整个应用被送进 shutdown 流程。当时的日志：

            asyncio.exceptions.CancelledError: Cancelled via cancel scope 240afccdc70
              by <Task pending name='Task-523' coro=<AgentHolder._rebuild_safely()>>
              File ".../starlette/routing.py", line 655, in lifespan

        表现极具迷惑性：重建计数正常增长、"可用工具 7 个"、没有不可用 Server，
        但此后所有工具调用永久 ClosedResourceError。也就是说"自愈代码把服务治死了"。
        请求路径只做 `session.call_tool`（跨 task 调用是允许的），不碰 enter/exit。
        """
        while True:
            try:
                cmd, fut = await asyncio.wait_for(
                    self._cmds.get(), timeout=self._health_interval)
            except asyncio.TimeoutError:
                try:
                    await self._health_check()      # 静默失效兜底：主动 ping
                except Exception as e:
                    log.error("健康巡检触发的重建失败（%s：%s），下轮再试",
                              type(e).__name__, e)
                continue

            if cmd == "stop":
                await self._close_sessions()
                if fut is not None and not fut.done():
                    fut.set_result(None)
                return

            try:
                await self._rebuild_with_retry()
                log.info("MCP 会话就绪（累计建立 %d 次，其中自愈 %d 次），可用工具 %d 个%s",
                         self._builds, self._recoveries, len(self._tools),
                         f"，未连上：{self._failures}" if self._failures else "")
            except BaseException as e:              # supervisor 绝不允许因单次失败而死掉
                log.error("MCP 会话重建失败（%s：%s）", type(e).__name__, str(e) or "无详情")
                if fut is not None and not fut.done():
                    fut.set_exception(e)
                    fut = None
            if fut is not None and not fut.done():
                fut.set_result(None)

    async def _rebuild_in_place(self) -> None:
        """重建整套会话与 Agent——只在 supervisor task 内被调用。

        **顺序必须是"先关旧、再建新"，不能反。** anyio 要求 cancel scope
        后进先出退出；先建新再关旧时，退出旧会话会抛

            RuntimeError: Attempted to exit a cancel scope that isn't
                          the current task's current cancel scope

        而且这个异常会把新会话一起搞死——表现极具迷惑性：重建计数正常增长、
        /health 显示"可用工具 7 个"、没有不可用 Server，但此后所有工具调用
        永久 ClosedResourceError。实测（_probe_rebuild.py）改成先关后建后，
        "健康状态下重建"与"杀掉子进程后重建"两个场景都正常。

        代价：重建期间有几秒没有可用 Agent。此时 ready 为 False，
        请求会快速拿到"正在重置连接"的提示，而不是撞在一个半死的会话上。
        """
        # 先摘掉 Agent 与旧会话（这个顺序就是 LIFO）
        was_ready = self._agent is not None      # 建立前是否已有可用 Agent → 决定算不算"自愈"
        self._agent = None
        self._tools = []
        await self._close_sessions()

        servers = load_servers_config()
        stack = AsyncExitStack()
        try:
            client = MultiServerMCPClient(servers)
            all_tools: list = []
            failures: list[str] = []
            sessions: dict = {}
            for name in servers:
                # 逐个连接：单个 Server 起不来只影响它自己，
                # 而不是像 get_tools() 那样"一个失败整批没有"
                try:
                    session = await stack.enter_async_context(client.session(name))
                    all_tools += await load_mcp_tools(session, server_name=name)
                    sessions[name] = session
                except Exception as e:
                    failures.append(name)
                    log.warning("MCP Server %s 连接失败（%s：%s），其工具本次不可用",
                                name, type(e).__name__, str(e).strip() or "无详情")

            tools = [t for t in all_tools if t.name not in HIDDEN_FROM_AGENT]
            if not tools:
                # 一个工具都没有的 Agent 只会对着用户胡编——宁可不启动
                raise RuntimeError("所有 MCP Server 均未连上，拒绝以空工具集启动 Agent")
            agent = create_react_agent(
                model=build_model(),
                tools=tools,
                prompt=SYSTEM_PROMPT,
                checkpointer=self._checkpointer,
            )
        except BaseException:
            await stack.aclose()          # 同一 task 内退出，顺序合法
            raise

        self._stack = stack
        self._agent = agent
        self._tools = tools
        self._failures = failures
        self._sessions = sessions
        self._ping_failures = {}
        self._note_build(was_ready)

    async def _rebuild_with_retry(self, attempts: int = 3) -> None:
        """建立失败就退避重试：会话建不起来等于服务不可用，不能干等 60 秒巡检。"""
        delay = 5.0
        for i in range(1, attempts + 1):
            try:
                await self._rebuild_in_place()
                return
            except Exception as e:
                if i == attempts:
                    raise
                log.warning("第 %d 次建立会话失败（%s：%s），%.0f 秒后重试",
                            i, type(e).__name__, str(e) or "无详情", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    def trigger_rebuild(self) -> None:
        """在请求路径上不阻塞地触发重建。

        本次请求已经失败了，没必要让用户再等一次重建（约 3–8 秒）：
        先把错误返回给前端，重建交给 supervisor，下一次重试就是好的。
        """
        self._cmds.put_nowait(("rebuild", None))

    async def _health_check(self) -> None:
        """主动 ping 各会话，连续两次失败才重建（只在 supervisor task 内被调用）。

        为什么需要：会话断在工具调用内部时，MCP 会把它包成
        ToolMessage(status="error") 交给模型自纠，异常不会冒到 API 层——
        这种"静默失效"抓不到异常，只能靠定期巡检兜住。

        为什么"连续两次"：会话内请求是串行的，若某次工具调用本身耗时长，
        ping 会在队列里排队，单次超时不足以判定会话已死。
        """
        if not self._sessions:
            return
        dead: list[str] = []
        for name, session in list(self._sessions.items()):
            try:
                await asyncio.wait_for(session.send_ping(), timeout=30)
                self._ping_failures[name] = 0
            except Exception as e:
                n = self._ping_failures.get(name, 0) + 1
                self._ping_failures[name] = n
                log.warning("会话 %s ping 失败第 %d 次（%s）", name, n, type(e).__name__)
                if n >= 2:
                    dead.append(f"{name}({type(e).__name__})")
        if dead:
            log.warning("健康巡检判定会话已失效：%s，触发重建", "、".join(dead))
            await self._rebuild_with_retry()    # 仍在 supervisor task 内，顺序合法

    async def _close_sessions(self) -> None:
        if self._stack is None:
            return
        stack, self._stack = self._stack, None
        try:
            await stack.aclose()
        except Exception as e:
            log.warning("关闭 MCP 会话时出错（已忽略）：%s", type(e).__name__)

    async def aclose(self) -> None:
        """通知 supervisor 关闭全部会话并退出。"""
        if self._task is None:
            await self._close_sessions()
            return
        try:
            await self._request("stop")
        except Exception as e:
            log.warning("停止会话管理 task 时出错（已忽略）：%s", type(e).__name__)
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None


@asynccontextmanager
async def create_app_agent():
    """便捷封装：给"跑一次就退出"的场景用（评测跑批、CLI、冒烟测试）。

    服务端请直接用 AgentHolder——它多了断线重建与健康巡检。
    """
    holder = AgentHolder()
    await holder.start()
    try:
        yield holder.agent, holder.tools
    finally:
        await holder.aclose()
