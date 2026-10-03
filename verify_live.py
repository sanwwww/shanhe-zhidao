#!/usr/bin/env python
"""对**已部署的线上服务**做端到端终验：走真实 HTTP/SSE，判定复用离线同一套 judge。

为什么不能直接用 evaluate.py 代替
--------------------------------
`evaluate.py` 在**进程内**直接跑 agent graph，绕过了整个 FastAPI 层。它和线上**不是**
同一个东西，差在三处，每一处都足以造成"离线全绿、线上有问题"：

1. **答案不是用户看到的那份**。线上有 `AnswerStreamFilter` 丢弃中间轮次前言
   （实测会输出 "I'll look this up in the knowledge base."）；evaluate.py 自己按
   "最后一次工具调用之后"拼接。判据是拿答案去判的，答案本身不同，结论就不必相同。
2. **没有传输层与生命周期**。SSE 断流、uvicorn lifespan、会话重建期间的 503
   `recovering`、限流 429——这些只在线上存在。
3. **跨进程**。本机跑得动不代表 Render 免费实例（0.1 CPU、冷启动 56s）跑得动。

第 3 点是真发生过的：2026-10-03 本地 125 tests 全绿 + 本机实跑 57/57，
但线上跑的还是 P0 版本，本地改动一行没上线——**"验过"和"验到线上"是两回事**。

判据必须是同一套：临时手写标准去测线上，等于每次换尺子量。

用法
----
    python verify_live.py --ids weather-01,halluc-02          # 指定用例
    python verify_live.py --category 天气                      # 按类别
    python verify_live.py --limit 6                            # 取前 6 条
    python verify_live.py --repeat 3 --ids weather-01          # 同题重复，看延迟分布

退出码：0 = 全部通过，1 = 有失败（可直接接 CI / 部署后门禁）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.scoring import judge, percentile, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent
TESTSET = ROOT / "evaluation" / "testset.json"
DEFAULT_BASE = "https://shanhe-zhidao.onrender.com"

RATE_LIMIT = 30          # 与 api_server.RATE_LIMIT 一致：每 IP 每分钟
REQ_TIMEOUT = 150.0      # 线上单题墙钟上限（服务端 90s 熔断 + SSE 传输余量）


def _load_cases(ids: str | None, category: str | None, limit: int | None) -> list[dict]:
    items = json.loads(TESTSET.read_text(encoding="utf-8"))
    if ids:
        want = [s.strip() for s in ids.split(",") if s.strip()]
        by_id = {i["id"]: i for i in items}
        missing = [w for w in want if w not in by_id]
        if missing:
            raise SystemExit(f"用例不存在：{', '.join(missing)}")
        items = [by_id[w] for w in want]
    if category:
        items = [i for i in items if i.get("category") == category]
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("筛选后没有任何用例")
    return items


async def warmup(client: httpx.AsyncClient, base: str) -> dict:
    """冷启动吸震：Render 免费实例休眠后首个请求可达 60s，别把它算进用例耗时。"""
    for attempt in range(1, 6):
        try:
            r = await client.get(f"{base}/health", timeout=120.0)
            if r.status_code == 200:
                return r.json()
            print(f"  [warmup {attempt}] HTTP {r.status_code} {r.text[:120]}")
        except Exception as e:                                    # noqa: BLE001
            print(f"  [warmup {attempt}] {type(e).__name__}: {e}")
        await asyncio.sleep(5)
    raise SystemExit("线上服务 5 次探活均失败，放弃终验")


class ServiceDown(RuntimeError):
    """线上服务不可用（不是模型答错）。

    必须和"用例失败"严格分开：把 502 计成"系统答错"，
    等于拿基础设施故障去否定模型质量——反过来也一样，故障会被掩码成好成绩。
    2026-10-03 实测吃过这个亏：45 条请求全 502、答案全空，
    其中三条纯反例用例因为"空答案不含任何被禁模式"被判了通过。
    """


class HealthGuard:
    """健康哨兵：把"服务还活着吗"和"模型答得对不对"分成两件事。

    除了探活，还盯 `/health` 的 session_builds：它中途变大 = 进程重启过
    （计数器是进程内存态，重启即归零后重新计数）。一旦发生，之前采到的
    延迟数据也全部作废，因为前后不是同一个进程实例。
    """

    def __init__(self, client: httpx.AsyncClient, base: str) -> None:
        self._client, self._base = client, base
        self.builds: int | None = None
        self.restarts = 0

    async def probe(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/health", timeout=60.0)
        except Exception:                                         # noqa: BLE001
            return False
        if r.status_code != 200:
            return False
        h = r.json()
        if h.get("unavailable_servers"):
            print(f"  ⚠ 有 Server 未连上：{h['unavailable_servers']}")
        builds = h.get("session_builds")
        if self.builds is not None and builds is not None and builds != self.builds:
            self.restarts += 1
            print(f"  ⚠ 检测到服务重启（session_builds {self.builds} → {builds}）")
        self.builds = builds if builds is not None else self.builds
        return True

    async def require_alive(self, context: str) -> None:
        """连续两次探活失败才判定服务已死，避免把一次抖动当成故障。"""
        for _ in range(2):
            if await self.probe():
                if self.restarts:
                    raise ServiceDown(f"{context}：服务在本次终验期间重启过（{self.restarts} 次）")
                return
            await asyncio.sleep(5)
        raise ServiceDown(f"{context}：线上服务连续两次探活失败（已不可用）")


async def ask(client: httpx.AsyncClient, base: str, message: str,
              thread_id: str) -> dict:
    """发一题，解析 SSE，返回用户实际看到的答案与工具调用序列。

    判定用的是 **AnswerStreamFilter 过滤后**的文本——也就是前端渲染的那份。
    这很关键：过滤前的中间轮次前言（"I'll look this up..."）不构成用户可见输出，
    拿它去判"有没有编数字"会误伤。
    """
    called: list[str] = []
    deltas: list[str] = []
    done: dict = {}
    error: str = ""
    status = 0
    t0 = time.time()

    async with client.stream("POST", f"{base}/api/chat/stream",
                             json={"message": message, "thread_id": thread_id},
                             timeout=REQ_TIMEOUT) as resp:
        status = resp.status_code
        if status != 200:
            body = (await resp.aread()).decode("utf-8", "replace")
            return {"called": [], "answer": "", "elapsed": round(time.time() - t0, 1),
                    "status": status, "error": f"HTTP {status}: {body[:200]}"}

        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event == "tool_start":
                    called.append(data.get("tool", ""))
                elif event == "delta":
                    deltas.append(data.get("text", ""))
                elif event == "done":
                    done = data
                elif event == "error":
                    error = data.get("answer", "")

    answer = "".join(deltas).strip() or error
    return {"called": called, "answer": answer,
            "elapsed": round(time.time() - t0, 1), "status": status,
            "error": "" if not error else error, "server_elapsed": done.get("elapsed"),
            "steps": done.get("steps"), "request_id": done.get("request_id")}


async def run_case(client: httpx.AsyncClient, base: str, item: dict,
                   rate: "RateGate", guard: "HealthGuard", repeat: int = 1) -> dict:
    """跑一条用例。传输层故障与模型答错在这里被彻底分开。

    状态码分三类的处理原则（这是被线上故障教出来的）：
      200        → 正常判定
      429 / 503  → 我们自己打太快 / 服务正在重建会话。退避重试；都不是"答错"
      5xx        → 基础设施故障。立刻探活：服务若已死就中止整轮终验——
                   继续跑只会把 502 一条条计成"模型失败"，把故障掩码成烂成绩
      其他 4xx   → 请求本身有问题，重试无意义，直接记录
    """
    last = None
    for _ in range(repeat):
        for attempt in range(4):
            await rate.wait()
            try:
                last = await ask(client, base, item["input"], f"live-{item['id']}-{attempt}")
            except Exception as e:                                # noqa: BLE001
                last = {"called": [], "answer": "", "elapsed": 0.0, "status": 0,
                        "error": f"{type(e).__name__}: {e}"}

            if last["status"] == 200:
                break
            if last["status"] in (429, 503):
                # 429 = 自己打太快；503 = 服务端正在重建会话。都不是"系统答错"
                print(f"  [{item['id']}] HTTP {last['status']}，退避后重试")
                await asyncio.sleep(20 if last["status"] == 429 else 8)
                continue
            if last["status"] >= 500 or last["status"] == 0:
                print(f"  [{item['id']}] HTTP {last['status']}，探活确认服务状态")
                await guard.require_alive(f"用例 {item['id']} 遇到 {last['status']}")
                await asyncio.sleep(8)
                continue
            break                      # 4xx：请求本身的问题，重试无意义
    assert last is not None
    verdict = judge(item, last["called"], last["answer"])
    return {**item, **last, **verdict}


class RateGate:
    """滑动窗口限速：服务端 30 req/min/IP，超了会 429——那是测量噪声，不是系统缺陷。"""

    def __init__(self, per_minute: int = RATE_LIMIT) -> None:
        self._per_minute = per_minute
        self._hits: list[float] = []

    async def wait(self) -> None:
        while True:
            now = time.time()
            self._hits = [t for t in self._hits if now - t < 60]
            if len(self._hits) < self._per_minute - 1:
                self._hits.append(now)
                return
            sleep = 61 - (now - self._hits[0])
            print(f"  [限速] 触达 {self._per_minute}/min，等 {sleep:.0f}s")
            await asyncio.sleep(max(sleep, 1))


def _reason(r: dict) -> str:
    """把失败原因说成人话——只说"FAIL"的回归报告等于没有报告。"""
    parts = []
    if not r.get("answer_nonempty", True):
        # 最先报这个：空答案会让所有纯反例用例"通过"，是最容易被误读的一种失败
        parts.append("答案为空（未作答，不是合规）")
    if not r["tools_ok"]:
        if r["missing_tools"]:
            parts.append(f"缺工具 {r['missing_tools']}")
        if r["illegal_tools"]:
            parts.append(f"误调 {r['illegal_tools']}")
    if not r["pos_ok"]:
        parts.append(f"必中未命中 {r['missing_must_match']}")
    if not r["kp_ok"]:
        parts.append(f"要点 {r['kp_hit']}/{r['kp_total']} < {r['kp_min']:.2f}")
    if not r["neg_ok"]:
        parts.append(f"踩反例 {r['violations']}")
    return "；".join(parts) or "—"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--ids", default=None, help="逗号分隔的用例 id")
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="每条用例重复次数（测延迟分布）")
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    cases = _load_cases(args.ids, args.category, args.limit)

    print(f"目标：{base}")
    print(f"用例：{len(cases)} 条 × {args.repeat} 次")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        health = await warmup(client, base)
        print(f"先验 /health：{json.dumps(health, ensure_ascii=False)}")
        if health.get("unavailable_servers"):
            print(f"⚠ 有 Server 未连上：{health['unavailable_servers']}——其工具不可用，"
                  f"相关用例必然失败，先查这个")

        rate = RateGate()
        guard = HealthGuard(client, base)
        guard.builds = health.get("session_builds")
        rows = []
        aborted = None
        for i, item in enumerate(cases, 1):
            try:
                r = await run_case(client, base, item, rate, guard, args.repeat)
            except ServiceDown as e:
                aborted = str(e)
                print(f"\n✗ 终验中止：{aborted}")
                print(f"  已完成 {len(rows)}/{len(cases)} 条。剩下的用例**没有测到**——"
                      f"既不算通过、也不算失败，是没测。")
                break
            rows.append(r)
            flag = "PASS" if r["passed"] else "FAIL"
            tools = ",".join(r["called"]) or "无工具"
            print(f"[{i}/{len(cases)}] {flag} {r['id']:<12} {r['elapsed']:>5.1f}s "
                  f"tools={tools}")
            if not r["passed"]:
                print(f"        ↳ {_reason(r)}")
                print(f"        答：{r['answer'][:200]}")

    summary = summarize(rows)
    elapsed = [r["elapsed"] for r in rows if r["status"] == 200]
    print("\n" + "=" * 62)
    print(f"总计 {summary['passed']}/{summary['total']} 通过"
          + (f"（另有 {len(cases) - len(rows)} 条未执行）" if aborted else ""))
    for cat, c in sorted(summary.get("by_category", {}).items()):
        print(f"  {cat:<6} {c['passed']}/{c['total']}")
    if guard.restarts:
        print(f"⚠ 本次终验期间服务重启过 {guard.restarts} 次，耗时数据不可比")
    if elapsed:
        print(f"端到端耗时  P50 {percentile(elapsed, 0.5):.2f}s  "
              f"P90 {percentile(elapsed, 0.9):.2f}s  max {max(elapsed):.2f}s")

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps({"base_url": base, "health": health,
                                   "aborted": aborted, "restarts": guard.restarts,
                                   "summary": summary, "rows": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {out}")

    if aborted:
        return 2          # 2 = 服务不可用，与"用例失败"(1) 区分开，便于接 CI 门禁
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
