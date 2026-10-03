# -*- coding: utf-8 -*-
"""
山河智导 · 评测判定与成本估算（纯函数，零外部依赖，可被单元测试直接调用）

为什么单独拆出来：
判定逻辑（工具断言 / 要点命中 / 反例断言 / token 成本）是评测里最容易写错、
也最该被测试覆盖的部分。把它和"调用 LLM"解耦后，这部分可以在零 token 成本下
用单测跑几千次，而真正的花钱跑批只负责"取数据"。
"""
import os
import re
import unicodedata

# ── 价格表（元/百万 token，可用环境变量覆盖）─────────────────────────
# 默认值按 DeepSeek 官方公开价目填写（缓存未命中输入 / 缓存命中输入 / 输出）。
# 价格会调整，所以不硬编码在逻辑里——报告里也会标注为"估算"。
DEFAULT_PRICE = {"in": 2.0, "in_cached": 0.5, "out": 8.0}


def estimate_price_from_env() -> dict:
    """从环境变量读价格，缺省回落到 DEFAULT_PRICE

    把单价做成可覆盖的：模型换了、官方调价了，评测结论不该跟着失真。
    """
    return {
        "in": float(os.getenv("PRICE_IN_PER_M", DEFAULT_PRICE["in"])),
        "in_cached": float(os.getenv("PRICE_IN_CACHED_PER_M", DEFAULT_PRICE["in_cached"])),
        "out": float(os.getenv("PRICE_OUT_PER_M", DEFAULT_PRICE["out"])),
    }


def text_of(content) -> str:
    """把消息 content 归一化成纯文本

    LangChain v1 下 content 可能是 str、content-blocks 列表（[{"type":"text","text":...}]）
    或带 .content 的对象。评测取答案必须走这一步，否则答案会变成 Python repr。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text") or "")
            else:
                parts.append(getattr(b, "text", "") or "")
        return "".join(parts)
    inner = getattr(content, "content", None)
    if inner is not None and inner is not content:
        return text_of(inner)
    return str(content)


def normalize(text: str) -> str:
    """断言前把文本归一化（NFKC），消除"同一个意思、不同写法"造成的假失败

    这类假失败已经在实测里出现过两次：
      · 温度：模型写 `21.3℃`（U+2103 单字符），而正则写的是 `°C`（两个字符）→ 误判"没给温度"
      · 日期：模型写 `Day 1`，而正则只认 `D1|第一天` → 误判"没给结构"
    两次都不是系统答错，是判据把"记法"当成了"内容"。

    用 NFKC 而不是逐个补正则：它是标准归一化，一次性覆盖全角字母数字、
    全角标点（：→:）、兼容字符（℃→°C、㎞→km）等一整类问题，
    而且规则写在代码里、可被单测钉住，不散落在 57 条用例的正则里。
    """
    return unicodedata.normalize("NFKC", text or "")


def judge(item: dict, called: list[str], answer: str) -> dict:
    """对单条用例做出通过判定

    判定由四个互相独立的检查组成，任一不过即整体不过：
      ① 工具断言：expected_tools 必须全被调用；forbidden_tools 一个都不能被调用
      ② 正面断言：keypoints 按 keypoint_min 比例命中；must_match 必须全部命中
      ③ 反例断言：must_not_match 命中任意一条即失败（对抗/幻觉用例的主判据）
      ④ 组合：passed = ① 且 ② 且 ③

    为什么要 ③：对抗与幻觉用例的正确行为是"拒绝 / 承认没收录"，
    用"必须命中某关键词"来表达既不可靠（拒绝的措辞千变万化），
    又容易被"先编一段再道歉"绕过去。反例断言直接钉死"不许出现什么"，
    判定稳得多。

    反例断言的经验（踩过坑才写下来）：**必须锚定到被问的那个对象**。
    最初 halluc-07/08 用的是通用反例（如 `(门票|票价).{0,8}\\d+\\s*元`），
    结果把"拒绝编造熊猫谷票价、但顺带引用了洋县朱鹮生态园的真实票价"
    判成了编造——那是合规回答。通用反例分不清"报了所问对象的价"和
    "报了别的对象的价"，所以改成对象锚定。
    """
    answer = normalize(answer)
    called = list(called or [])
    expected = list(item.get("expected_tools") or [])
    forbidden = list(item.get("forbidden_tools") or [])

    missing_tools = [t for t in expected if t not in called]
    illegal_tools = [t for t in forbidden if t in called]
    tools_ok = not missing_tools and not illegal_tools

    kps = list(item.get("keypoints") or [])
    hits = [kp for kp in kps if re.search(kp, answer)]
    kp_total, kp_hit = len(kps), len(hits)
    kp_min = float(item.get("keypoint_min", 0.6))
    kp_ratio = (kp_hit / kp_total) if kp_total else 1.0
    kp_ok = kp_ratio >= kp_min - 1e-9

    must = list(item.get("must_match") or [])
    missing_must = [p for p in must if not re.search(p, answer)]
    pos_ok = not missing_must

    negs = list(item.get("must_not_match") or [])
    violations = [p for p in negs if re.search(p, answer)]
    neg_ok = not violations

    return {
        "tools_ok": tools_ok,
        "missing_tools": missing_tools,
        "illegal_tools": illegal_tools,
        "kp_hit": kp_hit,
        "kp_total": kp_total,
        "kp_min": kp_min,
        "kp_ok": kp_ok,
        "pos_ok": pos_ok,
        "missing_must_match": missing_must,
        "neg_ok": neg_ok,
        "violations": violations,
        "passed": bool(tools_ok and kp_ok and pos_ok and neg_ok),
    }


def usage_from_messages(messages, price: dict | None = None) -> dict:
    """汇总一次调用里所有 LLM 的 token 用量并估算成本

    ReAct 一次问答会调用 LLM 多轮（每轮工具调用前后各一次），
    所以必须把 messages 里所有带 usage_metadata 的 AI 消息累加，
    只看最后一条会严重低估成本。
    """
    price = price or DEFAULT_PRICE
    inp = out = cached = 0
    llm_calls = 0
    for m in messages or []:
        um = getattr(m, "usage_metadata", None)
        if not um:
            continue
        llm_calls += 1
        inp += int(um.get("input_tokens") or 0)
        out += int(um.get("output_tokens") or 0)
        details = um.get("input_token_details") or {}
        cached += int(details.get("cache_read") or 0)
    fresh = max(0, inp - cached)
    cost = (fresh * price["in"] + cached * price["in_cached"] + out * price["out"]) / 1_000_000
    return {"input_tokens": inp, "output_tokens": out, "cached_tokens": cached,
            "total_tokens": inp + out, "llm_calls": llm_calls, "cost_cny": round(cost, 6)}


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数（样本量小时比 nearest-rank 更稳）"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def summarize(rows: list[dict]) -> dict:
    """整体 + 分类别汇总。分类别是必须的：总体通过率会把对抗/幻觉类的表现平均掉"""
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"total": 0, "passed": 0, "elapsed": [],
                                              "cost": 0.0, "tokens": 0})
        c["total"] += 1
        c["passed"] += 1 if r["passed"] else 0
        c["elapsed"].append(r["elapsed"])
        c["cost"] += r.get("cost_cny", 0.0)
        c["tokens"] += r.get("total_tokens", 0)
    elapsed = [r["elapsed"] for r in rows if not r.get("error")]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "tool_acc": (sum(1 for r in rows if r["tools_ok"]) / total) if total else 0.0,
        "errors": sum(1 for r in rows if r.get("error")),
        "avg_elapsed": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
        "p50": percentile(elapsed, 0.5),
        "p95": percentile(elapsed, 0.95),
        "max_elapsed": max(elapsed) if elapsed else 0.0,
        "total_tokens": sum(r.get("total_tokens", 0) for r in rows),
        "total_cost": sum(r.get("cost_cny", 0.0) for r in rows),
        "avg_cost": (sum(r.get("cost_cny", 0.0) for r in rows) / total) if total else 0.0,
        "by_category": by_cat,
    }
