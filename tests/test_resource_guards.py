# -*- coding: utf-8 -*-
"""资源守卫：把「进程被 OOM 杀死」这类故障钉死在测试里。

为什么单开一个文件
------------------
2026-10-03 线上验收抓到一次生产故障：知识库 MCP Server 依赖 ChromaDB +
onnxruntime（默认英文 embedding），单进程*峰值* RSS 271MB。它在 512MB 的
Render 免费实例上把整个服务打到 502，持续一分多钟，直到平台拉起新实例。

这个缺陷的特性决定了它必须单独防：

1. **功能用例全都测不出来**。检索结果完全正确，57 条评测也能全绿——
   崩的不是逻辑，是内存。本地开发机内存充裕，永远不会复现。
2. **单测/冒烟/评测都不会碰它**。它们在进程内跑，不叠加"主进程 + 4 个 MCP
   子进程"的真实占用。
3. **它只在生产规模下发生**。0.1 CPU / 512MB，还要和被杀的进程共享额度。

也就是说：**没有任何一个"跑一遍看结果对不对"的测试能发现它**。
唯一有效的防线是显式的资源断言——依赖白名单 + 峰值内存预算。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# 曾经把服务打死的依赖。列在这里的不是"不好"，而是"相对本项目收益不成比例"：
# 语料百条量级、关键词打分已是主信号，向量库的边际价值接近零，代价却是几百 MB 常驻内存。
BANNED_MODULES = ("chromadb", "onnxruntime", "torch", "transformers", "tensorflow")

# 单进程峰值 RSS 预算。修复前实测 271MB，修复后 64.6MB，取 150MB 留 2.3 倍余量：
# 正常改代码不会碰到，一旦有人再引入重型依赖立刻报警。
PEAK_RSS_BUDGET_MB = 150


def _imported_top_level_modules(path: Path) -> set[str]:
    """AST 提取一个 .py 文件里 import 的顶层模块名。

    用 AST 而不是文本搜索：故障复盘注释里会写 "ChromaDB"、"onnxruntime" 这些字，
    文本搜索会把说明文字误判成依赖——守卫测试自己先得是准的。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def _runtime_python_files() -> list[Path]:
    files = sorted((ROOT / "servers").glob("*.py"))
    files += [ROOT / f for f in ("agent_core.py", "api_server.py", "client.py") if (ROOT / f).exists()]
    return files


def test_runtime_modules_import_nothing_heavy():
    """运行时代码不得 import 重型依赖

    逐个 Server 都会作为独立子进程常驻，所以"哪个模块 import 了什么"直接等于内存账单。
    """
    offenders: dict[str, list[str]] = {}
    for path in _runtime_python_files():
        bad = sorted(_imported_top_level_modules(path) & set(BANNED_MODULES))
        if bad:
            offenders[path.name] = bad
    assert not offenders, (
        f"运行时代码引入了重型依赖 {offenders}。这些依赖会以独立子进程常驻内存，"
        f"在 512MB 实例上极易触发 OOM——若确需引入，请先给出内存实测数字并上调预算。")


def test_requirements_exclude_heavy_runtime_deps():
    """requirements.txt 不得包含重型依赖（真正的根因在依赖，不在 import）

    本次故障的入口正是这一行 `chromadb==1.5.9`：它拖进 onnxruntime、
    opentelemetry 等一整套东西。只堵 import 不堵依赖是治标。
    """
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pkgs = [l.split("==")[0].split(">=")[0].strip().lower() for l in lines if l and not l.startswith("#")]
    bad = [p for p in pkgs if p in BANNED_MODULES]
    assert not bad, f"requirements.txt 含重型依赖 {bad}：请确认它真的不可替代"


def _peak_rss_mb() -> float | None:
    """当前进程峰值 RSS（MB）。Linux 读 /proc 的 VmHWM，其他平台退回 psutil；都没有则 None。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1048576
    except Exception:                                             # noqa: BLE001
        return None


# 探针脚本：自包含，不 import 本测试模块（少一个耦合点，也避免 pytest 依赖渗进被测进程）。
# 峰值读取逻辑与 _peak_rss_mb 一致：/proc 优先，退回 psutil。
_PROBE = """
import asyncio, sys
sys.path.insert(0, {root!r})
import servers.knowledge_server as ks

asyncio.run(ks.search_knowledge("何尊为什么被称为镇馆之宝？"))
asyncio.run(ks.search_knowledge("乾陵门票和开放时间"))

def peak_mb():
    try:
        for line in open("/proc/self/status", encoding="utf-8"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    import psutil
    return psutil.Process().memory_info().peak_wset / 1048576

print("PEAK_MB", peak_mb())
"""


def test_knowledge_server_peak_memory_within_budget():
    """知识库 Server 跑完真实检索后，单进程峰值内存必须留在预算内。

    必须**另起子进程**测量：在 pytest 进程里量到的是 pytest + 全部测试依赖的峰值，
    和"线上这个 Server 子进程实际占多少"不是一回事，测了也没有意义。

    这条测试是本次线上故障唯一能自动拦住的防线——它不关心答案对不对，
    只关心"这个进程会不会把同机的兄弟进程一起拖进 OOM"。
    """
    if _peak_rss_mb() is None:
        pytest.skip("当前平台无法读取峰值 RSS（需 /proc 或 psutil）")

    probe = _PROBE.format(root=str(ROOT))
    proc = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"探针进程失败：{proc.stderr[-1500:]}"

    peak = None
    for line in proc.stdout.splitlines():
        if line.startswith("PEAK_MB"):
            peak = float(line.split()[1])
    assert peak is not None, f"探针未输出峰值内存：{proc.stdout[-500:]}"
    assert peak < PEAK_RSS_BUDGET_MB, (
        f"知识库 Server 峰值内存 {peak:.0f}MB 超出预算 {PEAK_RSS_BUDGET_MB}MB。\n"
        f"它作为独立 MCP 子进程与主进程共享实例内存，超限会被 OOM kill，\n"
        f"表现为整个服务 502（2026-10-03 已实际发生一次，峰值 271MB）。")
