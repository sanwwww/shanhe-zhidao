# -*- coding: utf-8 -*-
"""冒烟测试：python smoke_test.py [--full]
默认只测 4 个工具层（不调 LLM，零成本）；--full 加测一次真实 Agent 调用。"""
import asyncio
import sys


async def test_tools():
    from servers.weather_server import get_weather, get_clothing_advice
    from servers.guide_server import generate_guide
    from servers.knowledge_server import search_knowledge

    print("── 1. 天气工具（Open-Meteo 免 Key）")
    print((await get_weather("西安"))[:300], "\n")

    print("── 2. 穿搭建议")
    print(get_clothing_advice("秋"), "\n")

    print("── 3. 知识库 RAG（首次会下载 embedding 模型，约 79MB）")
    print(search_knowledge("何尊 中国一词最早的记载")[:400], "\n")

    print("── 4. 讲解词生成（儿童观众）")
    print(generate_guide("何尊", "儿童")[:400], "\n")


async def test_agent():
    import uuid
    from langchain_core.messages import HumanMessage
    from agent_core import create_app_agent

    print("── 5. Agent 全链路（真实调用 LLM）")
    async with create_app_agent() as (agent, tools):
        print("已接入工具：", [t.name for t in tools])
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="西安现在天气怎么样，适合出游吗？")]},
            {"configurable": {"thread_id": uuid.uuid4().hex[:12]}},
        )
        print("\nAgent 回答：\n", result["messages"][-1].content[:600])


if __name__ == "__main__":
    asyncio.run(test_tools())
    if "--full" in sys.argv:
        asyncio.run(test_agent())
