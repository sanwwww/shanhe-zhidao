# -*- coding: utf-8 -*-
"""
山河智导 · CLI 客户端（最小验证入口）
跑法：python client.py
"""
import asyncio
import uuid

from langchain_core.messages import HumanMessage

from agent_core import create_app_agent


async def run_chat_loop() -> None:
    async with create_app_agent() as (agent, tools):
        print("=" * 56)
        print("  山河智导 · 文旅导览 Agent（CLI 模式）")
        print(f"  已接入工具：{', '.join(t.name for t in tools)}")
        print("  输入 quit 退出")
        print("=" * 56)
        thread_id = uuid.uuid4().hex[:12]  # thread_id 决定记忆归属：同 id 即同一会话
        while True:
            try:
                msg = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("quit", "exit", ""):
                break
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=msg)]},
                {"configurable": {"thread_id": thread_id}},
            )
            reply = result["messages"][-1].content
            print(f"\n山河智导 > {reply}")


if __name__ == "__main__":
    asyncio.run(run_chat_loop())
