# -*- coding: utf-8 -*-
"""会话生命周期与断线自愈的守卫测试。

这些测试守的不是"功能有没有实现"，而是**几条用事故换来的顺序约束**——
它们被违反时的表现极具迷惑性：看起来一切正常，实际全部失效。
"""
import sys
from pathlib import Path

import anyio
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_core import AgentHolder, is_session_broken   # noqa: E402


def test_rebuild_must_close_old_sessions_before_opening_new():
    """重建必须先关旧、再建新。

    anyio 要求 cancel scope 后进先出退出。反过来做会抛
    `RuntimeError: Attempted to exit a cancel scope that isn't the current
    task's current cancel scope`，并且连带把新会话一起弄死——表现是
    "重建计数正常增长、/health 显示工具齐全、没有不可用 Server，
    但此后所有工具调用永久 ClosedResourceError"。

    这个坑实测踩过一次（自愈代码把服务治死了），所以用静态断言钉死顺序，
    不指望后来者从注释里读懂 it。
    """
    src = (ROOT / "agent_core.py").read_text(encoding="utf-8")
    body = src.split("async def _rebuild_in_place")[1].split("async def _rebuild_with_retry")[0]
    assert "self._close_sessions()" in body and "enter_async_context" in body
    assert body.index("self._close_sessions()") < body.index("enter_async_context"), (
        "重建顺序反了：必须先关旧会话再建新会话"
        "（anyio 的 cancel scope 必须后进先出退出）"
    )


def test_sessions_are_never_created_in_an_ephemeral_task():
    """会话的建立/关闭不能放在临时 task 里。

    第一版把重建放在 asyncio.create_task 里，退出旧会话时 anyio 取消了
    不属于该 task 的 cancel scope——而那个 scope 恰好是 uvicorn lifespan 所在的外层
    scope，直接把整个应用送进 shutdown 流程。所以 agent_core 里只允许为一个
    常驻 supervisor task 调用 create_task。

    用 AST 统计而不是字符串计数：注释和 docstring 里也会出现这个函数名。
    """
    import ast

    src = (ROOT / "agent_core.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create_task"
        and getattr(n.func.value, "id", "") == "asyncio"
    ]
    assert len(calls) == 1, f"只允许为常驻 supervisor 创建 task，实际有 {len(calls)} 处"
    assert 'name="mcp-session-supervisor"' in src


@pytest.mark.parametrize("exc", [
    anyio.ClosedResourceError(),      # 实测：stdio 子进程被 kill 后工具调用抛这个
    anyio.BrokenResourceError(),
    anyio.EndOfStream(),
    ConnectionError("boom"),
    BrokenPipeError("boom"),
    EOFError(),
])
def test_link_level_failures_are_recognized(exc):
    """链路级异常必须被判为"会话已断"，否则不会触发重建"""
    assert is_session_broken(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("门票字段缺失"),
    KeyError("city"),
    RuntimeError("业务逻辑错"),
])
def test_business_errors_do_not_trigger_rebuild(exc):
    """工具自身的业务错误不该触发整套会话重建，否则会被无谓打断"""
    assert is_session_broken(exc) is False


def test_nested_exception_group_and_cause_chain_are_unwrapped():
    """anyio 的 task group 会抛 ExceptionGroup，langgraph 可能再包一层。

    只看最外层类型会漏判，表现为"会话明明断了却不重建"。
    """
    grouped = ExceptionGroup("task group", [ValueError("无关"), anyio.ClosedResourceError()])
    assert is_session_broken(grouped) is True

    outer = RuntimeError("最后一轮")
    outer.__cause__ = ExceptionGroup("nested", [anyio.EndOfStream()])
    assert is_session_broken(outer) is True

    # 反过来：包里全是业务错误时不能误判
    assert is_session_broken(ExceptionGroup("g", [ValueError("a"), KeyError("b")])) is False


def test_holder_is_not_ready_before_start():
    """未启动 / 重建窗口内 ready 必须是 False。

    API 层据此快速返回 503，而不是把一个指向已关闭会话的 Agent 拿去撞墙。
    """
    holder = AgentHolder()
    assert holder.ready is False
    assert holder.tools == []
    assert holder.failures == []
    assert holder.rebuilds == 0
