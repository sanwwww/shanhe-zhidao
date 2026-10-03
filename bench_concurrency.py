# -*- coding: utf-8 -*-
"""
并发基线测量：定位"并发被串行化"到底发生在哪一层。

只测量、不改代码。三层各自独立入口：

  L1 direct   直接并发调用工具函数（绕过 MCP transport）——工具自身能否并行
  L2 mcp      经 MultiServerMCPClient 并发调用同一工具——MCP stdio 会话是否串行
  L3 agent    端到端并发（真 LLM + 真工具）——需 DEEPSEEK_API_KEY，会产生费用

判读方式（speedup = 串行总耗时 / 并发墙钟，理想值 = n）：
  speedup ≈ 1     该层完全串行，瓶颈就在这里
  speedup ≈ n     该层完全并行，瓶颈不在这里
  1 < speedup < n 部分并行，说明存在共享资源争用

跑法：
    python bench_concurrency.py --level direct --n 4
    python bench_concurrency.py --level mcp --n 4
    python bench_concurrency.py --level agent --n 2 --question "何尊为什么重要"
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_QUESTION = "何尊为什么重要"


async def _measure(one, n: int) -> dict:
    """先串行 n 次拿单次基线，再并发 n 次拿墙钟——两者用同一批调用路径。

    契约：one() 自己计时并返回本次耗时（秒），因此并发时每个协程报的是
    它自己的墙钟，能看出"排队"现象（各协程耗时被拉开即说明在争同一资源）。
    """
    serial = []
    for _ in range(n):
        serial.append(await one())

    t0 = time.perf_counter()
    conc = await asyncio.gather(*[one() for _ in range(n)])
    wall = time.perf_counter() - t0

    serial_total = sum(serial)
    return {
        "n": n,
        "serial_each": [round(x, 3) for x in serial],
        "serial_avg": round(serial_total / n, 3),
        "serial_wall": round(serial_total, 3),
        "conc_each": [round(x, 3) for x in conc],
        "conc_wall": round(wall, 3),
        "speedup": round(serial_total / wall, 2) if wall else 0.0,
    }


def _report(label: str, r: dict) -> None:
    n = r["n"]
    print(f"\n── {label} ───────────────────────────────")
    print(f"  单次耗时（串行 {n} 次）: {r['serial_each']}  均值 {r['serial_avg']}s")
    print(f"  串行总墙钟            : {r['serial_wall']}s")
    print(f"  并发 {n} 路各自耗时     : {r['conc_each']}")
    print(f"  并发总墙钟            : {r['conc_wall']}s")
    print(f"  speedup               : {r['speedup']}x  (理想 {n}x，1x = 完全串行)")
    if r["speedup"] >= n * 0.85:
        verdict = "完全并行 —— 瓶颈不在这一层"
    elif r["speedup"] <= 1.3:
        verdict = "完全串行 —— 瓶颈就在这一层"
    else:
        verdict = "部分并行 —— 存在共享资源争用"
    print(f"  判读                  : {verdict}")


async def bench_direct(n: int, question: str) -> dict:
    """L1：绕过 MCP transport，直接 await 工具函数。测工具自身（含 ChromaDB）能否并行。"""
    from servers.knowledge_server import search_knowledge

    async def one():
        t = time.perf_counter()
        await search_knowledge(question)
        return time.perf_counter() - t

    return await _measure(one, n)


async def bench_mcp(n: int, question: str, tool_name: str) -> dict:
    """L2：经 MultiServerMCPClient 调用工具。与线上链路一致，只是省掉 LLM。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from agent_core import load_servers_config

    client = MultiServerMCPClient(load_servers_config())
    tools = {t.name: t for t in await client.get_tools()}
    if tool_name not in tools:
        raise SystemExit(f"未找到工具 {tool_name}，可用：{sorted(tools)}")
    tool = tools[tool_name]

    # 先空跑一次：把子进程拉起、embedding 模型加载、语料缓存这些一次性成本排除在测量之外
    await tool.ainvoke({"query": question})

    async def one():
        t = time.perf_counter()
        await tool.ainvoke({"query": question})
        return time.perf_counter() - t

    return await _measure(one, n)


async def bench_session(n: int, question: str, server_name: str, tool_name: str) -> dict:
    """L2b：复用一条长期 session 加载工具。

    get_tools() 的官方 docstring 写明 "A new session will be created for each tool call"，
    因此 L2 每次调用都要重开子进程 + 重 import 依赖栈。本层验证复用会话能省掉多少。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from agent_core import load_servers_config

    client = MultiServerMCPClient(load_servers_config())
    async with client.session(server_name) as session:
        tools = {t.name: t for t in await load_mcp_tools(session, server_name=server_name)}
        tool = tools[tool_name]
        await tool.ainvoke({"query": question})   # 预热，排除首次开销

        async def one():
            t = time.perf_counter()
            await tool.ainvoke({"query": question})
            return time.perf_counter() - t

        return await _measure(one, n)


# 端到端验收问题集：覆盖"不调工具 / 单工具 / 多工具"三种形态，
# 用于改动前后的同题对比（跨题目比较 P50 没有意义，必须同题）。
E2E_QUESTIONS = [
    ("纯对话·不调工具", "你好，你能做什么"),
    ("知识检索·1 工具", "何尊为什么重要"),
    ("天气·1 工具", "西安明天天气怎么样"),
    ("路线·1 工具", "从钟楼到大雁塔怎么走"),
    ("复合行程·多工具", "兵马俑一日游怎么安排"),
]


async def bench_e2e(rounds: int) -> list[dict]:
    """对固定问题集做端到端调用，记录每题耗时与工具调用次数。"""
    import uuid

    from langchain_core.messages import HumanMessage, ToolMessage

    from agent_core import create_app_agent

    rows: list[dict] = []
    async with create_app_agent() as (agent, _tools):
        for label, q in E2E_QUESTIONS:
            for r in range(rounds):
                cfg = {"configurable": {"thread_id": f"e2e-{uuid.uuid4().hex[:8]}"},
                       "recursion_limit": 25}
                t = time.perf_counter()
                out = await agent.ainvoke({"messages": [HumanMessage(content=q)]}, cfg)
                dt = time.perf_counter() - t
                n_tool = sum(1 for m in out["messages"] if isinstance(m, ToolMessage))
                rows.append({"label": label, "question": q, "round": r + 1,
                             "elapsed": round(dt, 2), "tools": n_tool})
                print(f"  [{label}] {q} → {dt:.2f}s / {n_tool} 次工具")
    return rows


def _report_e2e(rows: list[dict]) -> None:
    import json
    import statistics

    print("\n── E2E 验收（固定问题集） ───────────────────")
    print(f"  {'类型':<20}{'问题':<22}{'耗时':>8}{'工具':>6}")
    for r in rows:
        print(f"  {r['label']:<20}{r['question']:<22}{r['elapsed']:>7.2f}s{r['tools']:>6}")
    e = [r["elapsed"] for r in rows]
    print(f"\n  均值 {statistics.mean(e):.2f}s　"
          f"P50 {statistics.median(e):.2f}s　"
          f"最大 {max(e):.2f}s　"
          f"工具调用合计 {sum(r['tools'] for r in rows)} 次")
    print("\n  JSONL（便于逐题留存对比）：")
    for r in rows:
        print("  " + json.dumps(r, ensure_ascii=False))


async def bench_agent(n: int, question: str) -> dict:
    """L3：端到端——真 LLM + 真工具。回答正确性不判定，只看延迟与并发行为。"""
    from langchain_core.messages import HumanMessage

    from agent_core import create_app_agent

    async with create_app_agent() as (agent, _tools):
        counter = {"i": 0}

        async def one():
            counter["i"] += 1
            cfg = {"configurable": {"thread_id": f"bench-{counter['i']}-{int(time.time())}"},
                   "recursion_limit": 25}
            t = time.perf_counter()
            await agent.ainvoke({"messages": [HumanMessage(content=question)]}, cfg)
            return time.perf_counter() - t

        return await _measure(one, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["direct", "mcp", "session", "agent", "e2e", "all"],
                    default="all")
    ap.add_argument("--n", type=int, default=4, help="并发路数")
    ap.add_argument("--rounds", type=int, default=1, help="E2E 每题重复轮数")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--tool", default="search_knowledge")
    ap.add_argument("--server", default="knowledge")
    args = ap.parse_args()

    print(f"问题：{args.question}")
    print(f"并发路数：{args.n}")

    if args.level in ("direct", "all"):
        _report("L1 direct（绕过 MCP，直接调工具函数）",
                asyncio.run(bench_direct(args.n, args.question)))
    if args.level in ("mcp", "all"):
        _report("L2 mcp（get_tools：每次调用新建会话）",
                asyncio.run(bench_mcp(args.n, args.question, args.tool)))
    if args.level in ("session", "all"):
        _report("L2b session（复用一条长期会话）",
                asyncio.run(bench_session(args.n, args.question, args.server, args.tool)))
    if args.level in ("agent", "all"):
        _report("L3 agent（端到端：真 LLM + 真工具）",
                asyncio.run(bench_agent(args.n, args.question)))
    if args.level in ("e2e", "all"):
        _report_e2e(asyncio.run(bench_e2e(args.rounds)))


if __name__ == "__main__":
    main()
