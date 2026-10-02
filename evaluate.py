# -*- coding: utf-8 -*-
"""
山河智导 · 效果评估器（方法论复用自 Ops Agent，面试王牌素材）
- 逐条真实调用 Agent（真 LLM + 真 MCP 工具）
- 三个指标：①工具选择准确率（期望工具全中）②结论要点命中率（正则，≥60% 及格）③耗时
- 产出：evaluation/evaluation_report.md + evaluation/evaluation_results.json（bad case 明细）
跑法：python evaluate.py [--limit N]
"""
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from agent_core import ROOT, create_app_agent

TESTSET = ROOT / "evaluation" / "testset.json"
REPORT = ROOT / "evaluation" / "evaluation_report.md"
RESULTS = ROOT / "evaluation" / "evaluation_results.json"


async def run_one(agent, item: dict) -> dict:
    t0 = time.time()
    try:
        result = await asyncio.wait_for(agent.ainvoke(
            {"messages": [HumanMessage(content=item["input"])]},
            {"configurable": {"thread_id": uuid.uuid4().hex[:12]}},
        ), timeout=120)  # 单例超时保险丝：LLM 服务端挂起时整条批次不被拖死
        msgs = result["messages"]
        called = [m.name for m in msgs if isinstance(m, ToolMessage)]
        answer = next((m.content for m in reversed(msgs)
                       if m.type == "ai" and m.content), "")
    except Exception as e:
        return {**item, "called": [], "answer": "", "elapsed": round(time.time() - t0, 1),
                "error": str(e), "tools_ok": False, "kp_hit": 0, "kp_total": len(item["keypoints"]),
                "passed": False}
    elapsed = round(time.time() - t0, 1)
    tools_ok = set(item["expected_tools"]).issubset(set(called))
    hits = [kp for kp in item["keypoints"] if re.search(kp, answer)]
    kp_hit, kp_total = len(hits), len(item["keypoints"])
    passed = tools_ok and (kp_total == 0 or kp_hit / kp_total >= 0.6)
    return {**item, "called": called, "answer": answer[:500], "elapsed": elapsed,
            "tools_ok": tools_ok, "kp_hit": kp_hit, "kp_total": kp_total, "passed": passed}


async def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    items = json.loads(TESTSET.read_text(encoding="utf-8"))[:limit]

    async with create_app_agent() as (agent, tools):
        print(f"评估集 {len(items)} 条，工具 {len(tools)} 个，开始跑批…", flush=True)
        rows = []
        for i, item in enumerate(items, 1):
            r = await run_one(agent, item)
            rows.append(r)
            mark = "✅" if r["passed"] else "❌"
            print(f"[{i}/{len(items)}] {mark} {r['id']} 工具{r['called']} "
                  f"要点{r['kp_hit']}/{r['kp_total']} {r['elapsed']}s", flush=True)

    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    tool_acc = sum(r["tools_ok"] for r in rows) / total * 100
    avg_t = sum(r["elapsed"] for r in rows) / total
    bad = [r for r in rows if not r["passed"]]

    lines = [
        "# 山河智导 · 效果评估报告",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"- 测试集：{total} 条（天气/路线/知识/讲解/复合 5 类）",
        f"- **通过率：{passed}/{total}（{passed/total*100:.0f}%）**",
        f"- 工具选择准确率：{tool_acc:.0f}%",
        f"- 平均耗时：{avg_t:.1f}s/例",
        "",
        "## Bad Case 明细" if bad else "## 全部通过，无 Bad Case",
    ]
    for r in bad:
        lines += [f"### {r['id']}（{r['category']}）",
                  f"- 输入：{r['input']}",
                  f"- 期望工具：{r['expected_tools']}，实际：{r['called']}",
                  f"- 要点命中：{r['kp_hit']}/{r['kp_total']}",
                  f"- 回答节选：{r.get('answer', '')[:200]}",
                  f"- 错误：{r.get('error', '无')}", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n通过率 {passed}/{total}，工具准确率 {tool_acc:.0f}%，均耗时 {avg_t:.1f}s")
    print(f"报告：{REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
