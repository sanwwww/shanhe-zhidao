# -*- coding: utf-8 -*-
"""
山河智导 · 讲解词生成 MCP Server
- 动机：来自项目作者在宝鸡青铜器博物院的讲解实习——同一件文物，讲给小学生和讲给历史爱好者，是两套话术
- 实现：从 knowledge/ 语料中提取目标条目 → 按观众类型输出"讲解要点 + 话术提示"（确定性输出，零额外 LLM 调用）
启动：python servers/guide_server.py
"""
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"

mcp = FastMCP("guide")

# 不同观众的讲解策略（来自一线讲解经验的沉淀）
AUDIENCE_TIPS = {
    "儿童": "👧 儿童团：多用提问互动（'猜猜这三个字念什么'）、讲动物纹饰和故事，避开生僻年代数字，每件文物控制在2分钟。",
    "学生": "🎓 学生团：结合课本考点（分封制、宗法制、青铜铸造工艺），点出'这个知识点考试可能出现'，引发抬头率。",
    "亲子": "👨‍👩‍👧 亲子团：给家长讲价值、给孩子讲故事，一条线同时照顾两代人；推荐互动环节（找铭文、数兽面）。",
    "老年": "👴 老年团：语速放慢、音量提高，多讲家国情怀和时代记忆，少走动、多找休息点。",
    "外宾": "🌍 外宾团：用类比翻译文化概念（铭文≈青铜上的'史书'，九鼎≈权力的'公章'），突出世界遗产维度。",
    "历史爱好者": "📜 深度游客：直接上学术细节——铭文释读、断代依据、考古发现经过，并推荐延伸阅读。",
    "通用": "🎙 通用讲解：三段式——是什么（年代/用途）→ 为什么重要（核心价值）→ 一个好记的故事。",
}


def _load_entries() -> list[tuple[str, str]]:
    """语料 → [(标题, 正文)]，与知识库切块逻辑一致（按 '## ' 标题）"""
    entries = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for chunk in re.split(r"(?m)^(?=## )", text):
            chunk = chunk.strip()
            if chunk.startswith("## ") and len(chunk) > 30:
                title = chunk.splitlines()[0].lstrip("# ").strip()
                entries.append((title, chunk))
    return entries


@mcp.tool()
def generate_guide(spot: str, audience: str = "通用") -> str:
    """为指定景点/文物生成针对特定观众类型的现场讲解词要点。用户要求"讲一讲/介绍一下/帮我组织讲解词/给XX人怎么讲"时使用。

    Args:
        spot: 景点或文物名，如 "何尊"、"兵马俑"、"华山"
        audience: 观众类型：通用 / 儿童 / 学生 / 亲子 / 老年 / 外宾 / 历史爱好者
    """
    entries = _load_entries()
    q = spot.strip()
    # 标题精确命中优先，其次正文包含
    hit = next(((t, b) for t, b in entries if q in t), None) or \
          next(((t, b) for t, b in entries if q in b), None)
    if not hit:
        known = "、".join(t.split("（")[0] for t, _ in entries[:12])
        return f"知识库暂未收录「{spot}」的讲解素材。已收录：{known} 等。"
    title, body = hit
    tip = AUDIENCE_TIPS.get(audience, AUDIENCE_TIPS["通用"])
    # 压缩正文为要点（按句号取前几句，保留信息密度最高的开头）
    sents = [s for s in re.split(r"(?<=[。！？])", body.split("\n", 1)[-1]) if s.strip()]
    digest = "".join(sents[:4])[:500]
    return (f"🎙 《{title}》讲解词要点（观众类型：{audience}）\n"
            f"📖 核心素材：{digest}\n"
            f"💡 观众策略：{tip}\n"
            f"⚠️ 讲解纪律：年代数字以展板为准，拿不准的不编造；互动提问后留 3 秒等待。")


@mcp.tool()
def list_guide_spots() -> str:
    """列出知识库中已收录讲解素材的全部景点/文物。用户问"你能讲哪些景点/文物"时使用。"""
    titles = [t for t, _ in _load_entries()]
    return "📚 已收录讲解素材：\n" + "\n".join(f"  · {t}" for t in titles)


if __name__ == "__main__":
    mcp.run(transport="stdio")
