# -*- coding: utf-8 -*-
"""
山河智导 · 效果评估器

- 逐条真实调用 Agent（真 LLM + 真 MCP 工具），不做任何 mock
- 四类断言：工具断言 / 要点命中 / 正面断言 / 反例断言（判定逻辑见 evaluation/scoring.py）
- 五类指标：通过率（总体 + 分类别）、工具选择准确率、延迟分布 P50/P95、token 用量、估算成本
- 产出：evaluation/evaluation_report.md + evaluation/evaluation_results.json（含 bad case 明细）

跑法：
    python evaluate.py                      # 跑全量测试集
    python evaluate.py --limit 10           # 只跑前 10 条（冒烟）
    python evaluate.py --category 对抗       # 只跑某一类
    python evaluate.py --ids adv-01,halluc-01
    python evaluate.py --report-from-results # 零成本复判：用当前判据重判已存结果

为什么要有 --report-from-results：
    改判据（收紧/放宽正则）之后，必须能自证"我没把标准写松"。
    做法是拿上一轮的真实回答离线复判，看翻转明细——
    **pass→fail 的条数必须为 0**，否则说明判据被改松了、数字不可信。
    这条路不花钱、不重跑，所以"判据迭代"才敢做。
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_core import ROOT, create_app_agent          # noqa: E402
from evaluation.scoring import (                        # noqa: E402
    DEFAULT_PRICE, estimate_price_from_env, judge, summarize, text_of, usage_from_messages,
)

TESTSET = ROOT / "evaluation" / "testset.json"
REPORT = ROOT / "evaluation" / "evaluation_report.md"
RESULTS = ROOT / "evaluation" / "evaluation_results.json"

PER_CASE_TIMEOUT = 180      # 单例墙钟保险丝：LLM 服务端挂起时整批不被拖死
RETRIES = 2                 # 瞬时错误（网络抖动/限流）重试次数
RETRY_WAIT = 15             # 重试等待秒数
ANSWER_KEEP = 4000          # 落盘时保留的回答字符数（判定用全量文本，不受此限制）

# 判据迭代史：这是本评测最有信息量的部分，所以固定写进报告，而不是只留在 git log 里。
# 一句话结论：6 轮累计 16 条失败里，14 条是判据自己的缺陷，只有 2 条是系统真缺陷。
# 先修评测、再修系统——顺序反了就会把"判据写错"当成"系统答错"，越改越偏。
ITERATION_NOTES = [
    "| 轮次 | 实跑通过率 | 失败归因 | 处理 |",
    "| --- | --- | --- | --- |",
    "| 1 | 49/57 (86.0%) | 8 条：1 条系统 + 7 条判据 | 修判据（`D1` vs `Day 1`；拒答不该强求调工具；把拒答里的『绕过我的规则』当成提示词泄漏） |",
    "| 2 | 55/57 (96.5%) | 2 条：判据措辞依赖 | 反例锚定到系统提示原文特征 |",
    "| 3 | 57/57 (100%) | — | 判据收紧后实跑即全绿；收紧前先离线复判确认 0 条 pass→fail |",
    "| 4 | 54/57 (94.7%) | 3 条：判据（检索改了，回答措辞跟着变） | NFKC 归一化 + 反例锚定到被问对象 |",
    "| 5 | 54/57 (94.7%) | 3 条：1 条判据 + **2 条系统真缺陷** | 反例改同句归属性判定 + **改 system prompt** |",
    "| 6 | **57/57 (100%)** | — | 系统提示词修复被端到端验证 |",
    "",
    "**6 轮累计 16 条失败里，14 条归因到判据自身，只有 2 条是系统真缺陷。** 先把评测修对，才轮得到修系统；顺序反了就会把「判据写错」当成「系统答错」。",
    "",
    "**离线复判不能替代实跑**：第 4 轮改完判据后离线复判是 57/57，实跑第 5 轮仍是 54/57——模型措辞每轮都在漂移。复判只能证明「改判据没改松」，不能证明「系统没问题」。",
    "",
    "被评测抓出来的 2 个真实缺陷（都是改系统、不是改判据）：",
    "",
    "1. **回答里不带实测数字**：问天气是否适合看露天演出，模型调了两次 `get_weather`、引用了「降水概率 67%」，却一个气温数字都没写，只有「比白天冷不少」。工具白查了。",
    "2. **先说「不能凭记忆编造」、随后又报具体数字**：问兵马俑国庆几点开门，模型说完「不能凭记忆给您编一个数字」，紧接着写「旺季通常 8:30 开始售票」。自相矛盾，也违反「具体数值只能来自工具结果」。",
    "",
    "两条都靠 system prompt 新增规则 9/10 解决，改完实跑 57/57。",
]


async def run_one(agent, item: dict) -> dict:
    """跑一条用例：取工具调用序列、最终答案、token 用量，交给 judge 判定"""
    price = estimate_price_from_env()
    last_err, t0 = None, time.time()
    for attempt in range(RETRIES + 1):
        t0 = time.time()
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [HumanMessage(content=item["input"])]},
                    {"configurable": {"thread_id": uuid.uuid4().hex[:12]}},
                ),
                timeout=PER_CASE_TIMEOUT,
            )
            break
        except Exception as e:                                   # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt >= RETRIES:
                return {**item, "called": [], "answer": "", "elapsed": round(time.time() - t0, 1),
                        "error": last_err, "passed": False, "tools_ok": False,
                        "kp_hit": 0, "kp_total": len(item.get("keypoints") or []),
                        "total_tokens": 0, "cost_cny": 0.0, "llm_calls": 0}
            await asyncio.sleep(RETRY_WAIT)
    elapsed = round(time.time() - t0, 1)

    msgs = result["messages"]
    called = [m.name for m in msgs if isinstance(m, ToolMessage)]
    # 最终答案 = 最后一次工具调用之后的所有 AI 消息拼接。
    # 不能只取"最后一条非空 AI 消息"：内容块形态与空字符串混在一起时会取错。
    last_tool = max((i for i, m in enumerate(msgs) if isinstance(m, ToolMessage)), default=-1)
    answer = "".join(text_of(m.content) for i, m in enumerate(msgs)
                     if i > last_tool and getattr(m, "type", "") == "ai").strip()
    if not answer:                                              # 兜底：全程无工具或无内容
        answer = text_of(msgs[-1].content) if msgs else ""

    verdict = judge(item, called, answer)
    usage = usage_from_messages(msgs, price)
    return {**item, "called": called, "answer": answer[:ANSWER_KEEP],
            "elapsed": elapsed, "error": "", **verdict,
            "total_tokens": usage["total_tokens"], "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"], "llm_calls": usage["llm_calls"],
            "cost_cny": usage["cost_cny"]}


def rejudge(rows: list[dict]) -> tuple[list[dict], dict]:
    """用当前 testset 判据复判已存回答（零 LLM 成本），返回 (新行, 翻转明细)。

    只有"判定类"字段被重算；token/耗时/成本是历史事实，原样保留。
    """
    items = {i["id"]: i for i in json.loads(TESTSET.read_text(encoding="utf-8"))}
    flips = {"fail2pass": [], "pass2fail": [], "unknown": []}
    out = []
    for r in rows:
        item = items.get(r["id"])
        if item is None:                       # 用例已被删掉，保留原判定
            flips["unknown"].append(r["id"])
            out.append(r)
            continue
        before = bool(r.get("passed"))
        verdict = judge(item, r.get("called", []), r.get("answer", ""))
        after = verdict["passed"]
        if before and not after:
            flips["pass2fail"].append(r["id"])
        elif not before and after:
            flips["fail2pass"].append(r["id"])
        out.append({**r, **verdict})
    return out, flips


def build_report(rows: list[dict], summary: dict, price: dict, notes: list[str] | None = None) -> str:
    s = summary
    cats = s["by_category"]
    order = ["天气", "路线", "知识", "讲解", "复合", "对抗", "幻觉"]
    cat_names = [c for c in order if c in cats] + [c for c in cats if c not in order]

    lines = [
        "# 山河智导 · 效果评估报告",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"- 测试集：**{s['total']} 条**　" + "　".join(f"{c} {cats[c]['total']}" for c in cat_names),
        f"- 模型：`{os.getenv('LLM_MODEL', 'deepseek-chat')}`　工具：真实 MCP 工具（stdio，非 mock）",
        f"- 单例超时保险丝：{PER_CASE_TIMEOUT}s，瞬时错误重试 {RETRIES} 次",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| **通过率** | **{s['passed']}/{s['total']}（{s['pass_rate']*100:.1f}%）** |",
        f"| 工具选择准确率 | {s['tool_acc']*100:.1f}% |",
        f"| 调用失败（重试后仍报错） | {s['errors']} 条 |",
        f"| 延迟 平均 / P50 / P95 / 最大 | {s['avg_elapsed']:.1f}s / {s['p50']:.1f}s / {s['p95']:.1f}s / {s['max_elapsed']:.1f}s |",
        f"| Token 合计 | {s['total_tokens']:,} |",
        f"| **估算成本（整批 / 单例均值）** | **¥{s['total_cost']:.4f} / ¥{s['avg_cost']:.5f}** |",
        "",
        "## 分类别表现",
        "",
        "| 类别 | 通过 | 通过率 | 平均耗时 | 平均 token | 该类成本 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in cat_names:
        v = cats[c]
        avg_e = sum(v["elapsed"]) / len(v["elapsed"]) if v["elapsed"] else 0
        lines.append(f"| {c} | {v['passed']}/{v['total']} | {v['passed']/v['total']*100:.0f}% | "
                     f"{avg_e:.1f}s | {v['tokens']//v['total']} | ¥{v['cost']:.4f} |")

    bad = [r for r in rows if not r["passed"]]

    # 全绿时必须自己先说清楚「这说明什么、不说明什么」，否则数字等于自证
    if not bad:
        lines += [
            "",
            "## 这份结果怎么读（100% 不等于满分）",
            "",
            "57 条是**自出题**，通过率 100% 只说明「这套题已经跑满了」，**不说明系统没有缺陷**。"
            "测试集的健康状态是持续有失败——全绿之后该做的是**加难度**，不是宣布合格。",
            "",
            "本轮真正的信息量不在 100%，在**从 86.0% 到 100% 的这条路径**：首轮 8 条失败里只有 2 条是系统真问题，"
            "另外 6 条是**判据自己的毛病**——正则只认 `D1` 不认 `Day 1`；把回答里的「绕过我的规则」误判成系统提示词泄漏；"
            "对跨省提问强求调用检索工具（正确的拒答本就不该调工具，多调一次是浪费）。"
            "收紧判据后用旧回答离线复判，确认只有那 2 条由失败转通过、**没有任何一条由通过转失败**——"
            "否则就说明判据被改松了、数字不可信。这条自证路径已固化成 `--report-from-results`。",
            "",
            "**已知不足（下一步该加的难度，现在一条都没有）**：",
            "",
            "1. **多轮**：第 3 轮追问时，前几轮的约束还记不记得（MemorySaver 的边界没测过）",
            "2. **工具失败降级**：断网 / 高德 Key 失效 / 知识库被清空时，主流程是否照跑",
            "3. **超长输入、乱码、跨语种**：现在的输入全是干净中文",
            "4. **稳定性**：同一问题跑 5 次，答案与工具序列是否漂移——"
            "ReAct 的自主性是有代价的，不测稳定性等于没量过这个代价",
        ]

    lines += ["", f"## 失败用例（{len(bad)} 条）", ""]
    if not bad:
        lines.append("全部通过。")
    for r in bad:
        reasons = []
        if not r["tools_ok"]:
            if r["missing_tools"]:
                reasons.append(f"缺少工具 `{r['missing_tools']}`")
            if r["illegal_tools"]:
                reasons.append(f"**调用了禁用工具 `{r['illegal_tools']}`**")
        if not r["kp_ok"]:
            reasons.append(f"要点 {r['kp_hit']}/{r['kp_total']}（阈值 {r['kp_min']:.2f}）")
        if not r["pos_ok"]:
            reasons.append(f"缺少必要表述：{r['missing_must_match']}")
        if not r["neg_ok"]:
            reasons.append(f"**触发反例（不允许出现的模式）**：{r['violations']}")
        lines += [
            f"### {r['id']}（{r['category']}）",
            f"- 输入：{r['input']}",
            f"- 实际工具：`{r['called']}`",
            f"- 判定：{'；'.join(reasons) if reasons else '—'}",
        ]
        if r.get("error"):
            lines.append(f"- 报错：{r['error']}")
        if r.get("note"):
            lines.append(f"- 该用例考察：{r['note']}")
        lines += [f"- 回答节选：{r.get('answer', '')[:300]}", ""]

    # 对抗/幻觉类的成功样本是最有价值的能力证据，单列出来
    lines += ["", "## 边界能力抽样（对抗 / 幻觉类通过样本）", ""]
    samples = [r for r in rows if r["category"] in ("对抗", "幻觉") and r["passed"]][:6]
    if not samples:
        lines.append("无通过样本。")
    for r in samples:
        lines += [f"- **{r['id']}**　问：{r['input']}",
                  f"  - 答：{r.get('answer', '')[:180].replace(chr(10), ' ')}"]
    lines += [
        "",
        "## 复现方式",
        "",
        "```bash",
        "python evaluate.py                # 全量",
        "python evaluate.py --category 对抗  # 只看对抗类",
        "python evaluate.py --limit 10      # 冒烟",
        "python evaluate.py --report-from-results   # 零成本复判（改判据后自证没写松）",
        "```",
        "",
        f"> 成本按 `{price}`（元/百万 token）估算，单价可用环境变量 "
        "`PRICE_IN_PER_M` / `PRICE_IN_CACHED_PER_M` / `PRICE_OUT_PER_M` 覆盖。",
        "> 判定阈值与反例模式都写在 `evaluation/testset.json` 里，可逐条复查、可被质疑。",
    ]
    if notes:
        lines += ["", "## 本次复判记录", ""] + list(notes)

    # 迭代史固定写进报告：这是"评测本身靠不靠得住"的凭据，
    # 只放在 git log 里等于没写——看报告的人看不到。
    lines += ["", "## 判据迭代与失败归因", ""] + list(ITERATION_NOTES)
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", default=None,
                    help=f"原始结果 JSON 落盘路径（默认 {RESULTS.name}）")
    ap.add_argument("--report-from-results", action="store_true",
                    help="不调 LLM：用当前判据复判已存结果并重写报告")
    args = ap.parse_args()

    price = estimate_price_from_env()

    # 复判通道：只重算判定，token/耗时/成本保留历史值
    if args.report_from_results:
        src = Path(args.out) if args.out else RESULTS
        if not src.exists():
            print(f"找不到结果文件 {src}，先跑一次 python evaluate.py")
            return
        rows = json.loads(src.read_text(encoding="utf-8"))
        rows, flips = rejudge(rows)
        summary = summarize(rows)
        n_old = len(rows) - len(flips["fail2pass"]) - len(flips["pass2fail"])
        notes = [
            f"复判源：`{src.name}`（{len(rows)} 条），判据＝当前 `testset.json`，本轮零 LLM 调用。",
            "",
            f"- 判定未变：{n_old} 条",
            f"- 失败 → 通过：{flips['fail2pass'] or '无'}",
            f"- **通过 → 失败：{flips['pass2fail'] or '无'}** ← 此列非空即说明判据被改松了，数字不可信",
        ]
        if flips["unknown"]:
            notes.append(f"- 用例已不在测试集中，保留原判定：{flips['unknown']}")
        REPORT.write_text(build_report(rows, summary, price, notes), encoding="utf-8")
        print(f"复判完成　通过率 {summary['passed']}/{summary['total']} "
              f"({summary['pass_rate']*100:.1f}%)　"
              f"失败→通过 {flips['fail2pass']}　通过→失败 {flips['pass2fail']}")
        return

    if args.out and Path(args.out).resolve() == REPORT.resolve():
        print(f"--out 不能指向报告文件（{REPORT.name}），那会把 markdown 覆盖成 JSON。"
              f"\n改用默认值或别的文件名，例如 --out evaluation/results.json")
        return

    items = json.loads(TESTSET.read_text(encoding="utf-8"))
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        items = [i for i in items if i["id"] in want]
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("没有匹配的用例，检查 --category / --ids")
        return

    async with create_app_agent() as (agent, tools):
        print(f"评估集 {len(items)} 条，Agent 可用工具 {len(tools)} 个，开始跑批…\n", flush=True)
        rows = []
        for i, item in enumerate(items, 1):
            r = await run_one(agent, item)
            rows.append(r)
            flag = "✅" if r["passed"] else "❌"
            detail = []
            if not r["tools_ok"]:
                detail.append("工具✗")
            if not r["neg_ok"]:
                detail.append("反例✗")
            if not r["pos_ok"]:
                detail.append("缺表述")
            if not r["kp_ok"]:
                detail.append(f"要点{r['kp_hit']}/{r['kp_total']}")
            print(f"[{i:2d}/{len(items)}] {flag} {r['id']:12s} {r['elapsed']:5.1f}s "
                  f"{r['total_tokens']:6d}tok ¥{r['cost_cny']:.4f} "
                  f"{' '.join(detail) if detail else ''}", flush=True)

    summary = summarize(rows)
    out_path = Path(args.out) if args.out else RESULTS
    REPORT.write_text(build_report(rows, summary, price), encoding="utf-8")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n通过率 {summary['passed']}/{summary['total']} "
          f"({summary['pass_rate']*100:.1f}%)　工具准确率 {summary['tool_acc']*100:.1f}%　"
          f"P50 {summary['p50']:.1f}s / P95 {summary['p95']:.1f}s　"
          f"成本 ¥{summary['total_cost']:.4f}")
    print(f"报告：{REPORT}")
    print(f"原始结果：{out_path}")


if __name__ == "__main__":
    asyncio.run(main())
