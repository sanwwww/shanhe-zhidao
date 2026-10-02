# -*- coding: utf-8 -*-
"""
山河智导 · 文旅知识库 MCP Server（RAG-as-MCP-Tool）
- 把 RAG 检索封装成一个标准 MCP 工具：对 Agent 来说，知识库和天气、地图是同构的工具
- 复用 Ops Agent 验证过的方案：ChromaDB + 混合检索（向量召回 6 条 → 关键词重排取 top3）
- 空库自愈：首次检索发现库为空自动入库（部署到 Render 后容器磁盘会重置，必须有保险丝）
启动：python servers/knowledge_server.py
"""
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "shanhe_kb"
MAX_CHARS = 1200

mcp = FastMCP("knowledge")

_client = None
_col = None


def _collection():
    """懒加载 ChromaDB（首次调用才载入 embedding 模型，加快 Server 启动）"""
    global _client, _col
    if _col is None:
        import chromadb
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _col = _client.get_or_create_collection(COLLECTION)
        if _col.count() == 0:
            _build(_col)  # 空库自愈保险丝
    return _col


def _build(col) -> int:
    """重建式入库：按 '## ' 二级标题切块（一个标题=一个完整知识点，语义不断裂）"""
    docs, ids, metas = [], [], []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, chunk in enumerate(re.split(r"(?m)^(?=## )", text)):
            chunk = chunk.strip()
            if len(chunk) < 30 or not chunk.startswith("## "):
                continue
            title = chunk.splitlines()[0].lstrip("# ").strip()
            docs.append(chunk)
            ids.append(f"{md.stem}-{i}")
            metas.append({"source": md.name, "title": title})
    if docs:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def _tokens(q: str) -> list[str]:
    """查询关键词：英文/数字 token + 中文 2-gram（针对 MiniLM 英文模型中文召回差的补丁）"""
    toks = re.findall(r"[A-Za-z0-9]+", q)
    zh = re.findall(r"[一-龥]+", q)
    for seg in zh:
        toks += [seg[i:i + 2] for i in range(len(seg) - 1)] or ([seg] if seg else [])
    return toks


@mcp.tool()
def search_knowledge(query: str) -> str:
    """检索陕西文旅知识库（景点背景、文物故事、参观攻略、行程方法论、美食贴士）。任何涉及景点介绍、文物历史、游玩攻略、行程规划建议的问题，必须先调用本工具获取知识再回答，禁止凭模型记忆作答。

    Args:
        query: 检索词，如 "何尊 中国一词"、"兵马俑 参观攻略"、"华山 路线"
    """
    try:
        col = _collection()
        if col.count() == 0:
            return "知识库暂时为空，请基于通用知识回答，并提示用户答案未经知识库校验。"
        res = col.query(query_texts=[query], n_results=6)  # 向量过召回
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        toks = _tokens(query)
        scored = []
        for doc, meta, dist in zip(docs, metas, dists):
            hits = sum(1 for t in toks if t and t in doc)
            score = hits * 2 + max(0.0, 1.0 - dist)  # 关键词命中加权 ×2 + 向量相似度
            scored.append((score, doc, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, doc, meta in scored[:3]:
            out.append(f"【{meta['title']}】（来源：{meta['source']}）\n{doc}")
        return ("\n\n---\n\n".join(out))[:MAX_CHARS]
    except Exception as e:
        # RAG 是增强不是依赖：检索挂了返回提示语，Agent 主流程继续
        return f"知识库检索异常（{e}），请基于通用知识谨慎回答。"


@mcp.tool()
def rebuild_knowledge() -> str:
    """重建知识库索引（管理员操作用，普通用户问题不要调用本工具）。"""
    try:
        col = _collection()
        n = _build(col)
        return f"知识库已重建，共 {n} 条知识。"
    except Exception as e:
        return f"重建失败：{e}"


if __name__ == "__main__":
    if "--build" in sys.argv:  # 允许 python servers/knowledge_server.py --build 手动建库
        _collection()
        print(f"知识库就绪，共 {_col.count()} 条")
    else:
        mcp.run(transport="stdio")
