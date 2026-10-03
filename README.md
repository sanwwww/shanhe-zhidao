# 山河智导 · 文旅导览 Agent

> 基于 **MCP（Model Context Protocol）** 与 **LangGraph ReAct** 的多工具文旅导览 Agent：
> 一句话完成「查路线 + 看天气 + 听讲解 + 规划行程」。
> 项目动机来自作者在宝鸡青铜器博物院的讲解实习。

## 架构

```
用户（Web 前端，SSE 流式）
   │  POST /api/chat/stream  {message, thread_id}
   ▼
FastAPI 服务层（api_server.py，lifespan 管理连接生命周期）
   ▼
LangGraph ReAct Agent（create_react_agent + checkpointer 多轮记忆）
   │  MultiServerMCPClient 统一接入 ↓（stdio）
   ├── weather   天气/穿搭（Open-Meteo 免 Key，和风可选）
   ├── amap      高德路线规划/场所搜索（AMAP_KEY 可选，无 Key 优雅降级）
   ├── knowledge 陕西文旅知识库 ★RAG-as-MCP-Tool（ChromaDB + 混合检索，空库自愈）
   └── guide     讲解词生成（按观众类型：儿童/学生/亲子/老年/外宾/历史爱好者）
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 DEEPSEEK_API_KEY（必填）
python servers/knowledge_server.py --build   # 预建知识库（首次会下载约 79MB embedding 模型）

python client.py            # CLI 模式
python api_server.py        # Web 模式 → http://127.0.0.1:8001
```

## 质量保障

```bash
python -m pytest tests/ -v      # 单元测试 18 项（零 LLM 成本）
python smoke_test.py            # 工具层冒烟（零 LLM 成本）
python smoke_test.py --full     # +一次真实 Agent 全链路
python evaluate.py              # 30 条评估集跑批（真实 LLM，约几分钟）
```

## 设计要点（面试口径）

- **RAG 即 MCP 工具**：教程常见口径是"用 MCP 就不用 RAG"——本项目把知识库检索封装成独立 MCP Server，对 Agent 来说与天气、地图同构；检索挂了主流程照跑（增强不是依赖）。
- **MCP 的价值演示**：加一个工具 = `servers_config.json` 加一段 JSON，主代码零改动。
- **混合检索**：英文 MiniLM 中文召回差 → 向量召回 6 条 + 关键词重排取 top3（复用自 Ops Agent）。
- **安全**：工具全只读；公网限流 30 次/分/IP；消息 ≤500 字；Key 只走环境变量；管理员类工具（`rebuild_knowledge`）在代码层从 Agent 工具列表剔除——权限靠代码边界，不靠 prompt 约束模型。
- **可靠性**：单次请求 90s 墙钟熔断；ReAct 步数上限 12；客户端断开即停止生成（不浪费 token）；异常详情只进服务端日志，不回传堆栈给用户。
- **流式答案过滤**：ReAct 每轮工具调用前模型都会输出一段“前言”（实测会输出英文句子），按 `run_id` 分段识别并整段丢弃，只把最终答案段推给用户。不用长度阈值判断——实测前言可达 57 字符，阈值法无法可靠区分。
- **边界**：无订票/预约能力；模型决策、代码执行；单 Agent 循环（不含任务规划/多 Agent）。
