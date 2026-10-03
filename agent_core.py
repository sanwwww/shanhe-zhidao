# -*- coding: utf-8 -*-
"""
山河智导 · Agent 组装核心（被 client.py / api_server.py / evaluate.py 共用）
- MultiServerMCPClient：按 servers_config.json 拉起全部 MCP Server，自动发现工具
- create_react_agent：LangGraph 预建 ReAct 循环（= Ops Agent 里手写的那个循环的框架封装版）
- checkpointer：多轮记忆，thread_id 隔离会话
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """你是"山河智导"，一位专业的陕西文旅导览 Agent，由一位有博物院讲解经验的开发者打造。

工作规则：
1. 先判断用户意图类型：路线交通 / 天气穿搭 / 景点文物知识 / 讲解词 / 行程规划（复合型）。
2. 景点、文物、攻略、行程类问题，必须先用 search_knowledge 检索知识库再回答，禁止凭模型记忆编造年代、数字、票价。
3. 路线问题用 plan_route，场所位置用 search_place，天气用 get_weather，穿搭建议配合 get_clothing_advice。
4. 用户要求"讲一讲/组织讲解词"时用 generate_guide，并主动询问观众类型（儿童/学生/亲子/老年/外宾/历史爱好者）。
5. 复合型行程规划（如"一天怎么玩"）应组合调用：知识库（景点背景）+ 天气 + 路线，给出结构化方案。
6. 结论结构化：先给直接答案，再给依据（注明来自知识库/实时查询），最后给一个实用的下一步建议。
7. 工具返回"未配置/查询失败"时，诚实告知并给出替代方案，不要假装查到了。
8. 全部工具都是只读查询，你不具备订票、预约的能力，涉及预约要引导用户去官方公众号。"""


def load_servers_config() -> dict:
    """读 servers_config.json 并把相对路径解析成绝对路径（工作目录无关，更稳）"""
    cfg = json.loads((ROOT / "servers_config.json").read_text(encoding="utf-8"))
    for name, s in cfg.items():
        if s.get("transport") == "stdio":
            s["args"] = [str((ROOT / a).resolve()) if a.startswith("servers/") else a
                         for a in s["args"]]
            # 必须用当前解释器拉起 Server（配置里的 "python" 可能指向无依赖的系统 Python）
            s["command"] = sys.executable
            # MCP stdio 默认只转发白名单环境变量（PATH 等），业务密钥必须显式注入子进程；
            # 显式传 env 时会整体替换而非合并，所以先复制完整环境再注入
            amap_key = os.getenv("AMAP_KEY")
            if amap_key:
                merged = dict(os.environ)
                merged["AMAP_KEY"] = amap_key
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


@asynccontextmanager
async def create_app_agent():
    """异步上下文：连接 MCP → 取工具 → 建 Agent；退出时清理连接（FastAPI lifespan 同款思路）"""
    client = MultiServerMCPClient(load_servers_config())
    try:
        all_tools = await client.get_tools()
        tools = [t for t in all_tools if t.name not in HIDDEN_FROM_AGENT]
        checkpointer = MemorySaver()  # 生产可换 SqliteSaver/PostgresSaver，接口不变
        agent = create_react_agent(
            model=build_model(),
            tools=tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        yield agent, tools
    finally:
        # langchain-mcp-adapters 0.3：get_tools 管理会话生命周期，无需显式 close
        pass
