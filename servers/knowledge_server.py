# -*- coding: utf-8 -*-
"""
山河智导 · 文旅知识库 MCP Server（RAG-as-MCP-Tool）
- 把 RAG 检索封装成一个标准 MCP 工具：对 Agent 来说，知识库和天气、地图是同构的工具
- 检索策略：全量打分（标题命中 ≫ 正文关键词命中 > 向量相似度），取 top3
  为什么不做"向量 topN → 重排"：ChromaDB 默认 embedding 是英文模型 all-MiniLM-L6-v2，
  对中文的语义区分度极差，实测向量 top10 里常常没有正确条目，重排再准也救不回来。
  语料在百条量级，全量扫描成本可忽略（约 0.1-0.3s），召回有保证。
- 空库自愈：首次检索发现库为空自动入库（部署到 Render 后容器磁盘会重置，必须有保险丝）
启动：python servers/knowledge_server.py
"""
import math
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "shanhe_kb"
MAX_CHARS = 2000   # 返回给模型的检索片段上限：知识库扩到 90+ 条后单条最长约 370 字，
                   # 3 条拼接最大约 1100 字；留到 2000 是为了不截断
TOP_K = 3          # 最终返回给模型的条目数（打分已按"标题命中 ≫ 正文命中 > 向量"排序）
MIN_SPAN = 2       # 相关度下限：标题与问题至少要共享 2 个连续汉字，才认为"问到了这个条目"

mcp = FastMCP("knowledge")

_client = None
_col = None
_docs = None       # 全量语料内存缓存（检索在全量上打分，见 _all_docs）
_idf_cache = None  # 语料级 IDF 缓存（打分用，见 _idf）


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
    """重建式入库：按 '## ' 二级标题切块（一个标题=一个完整知识点，语义不断裂）

    先清空再写入（而不是 upsert）：upsert 只会新增/覆盖同 id 条目，
    一旦知识库删改了标题（id 变化），旧块会以孤儿身份留在库里，
    检索时可能召回早已删除的内容。重建就该是重建。
    """
    global _docs, _idf_cache
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
    old = col.get(include=[])["ids"]
    if old:
        col.delete(ids=old)
    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    _docs = None          # 缓存失效，下次检索重新拉全量
    _idf_cache = None     # IDF 依赖语料统计，必须一起失效
    return len(docs)


# 低信息量填充词：只从"查询"和"用于比对的标题"里剥离，不动语料正文。
# 不加这一步会出真 bug："华山夜爬怎么安排" 与《回民街、洒金桥与永兴坊怎么选》
# 共享"怎么"这一 2 字片段，最长公共子串打分把它抬到和"华山"同分，
# 通用条目的正文又更啰嗦，结果抢走了首位。剥离填充词后这类假重合直接归零。
_FILLER = ("为什么", "什么时候", "怎么", "什么", "时候", "如何", "哪里", "哪儿",
           "多少", "哪些", "哪个", "请问", "帮我", "告诉我", "介绍一下", "介绍",
           "一下", "可以", "需要", "值得", "推荐", "安排", "最好", "还有",
           "就是", "以及", "或者", "是不是", "有没有")


def _strip_filler(text: str) -> str:
    for w in _FILLER:
        text = text.replace(w, "")
    return text


def _tokens(q: str) -> list[str]:
    """查询关键词：英文/数字 token + 中文 2-gram 与 3-gram（先剥填充词）

    中文 3-gram 用于提升精度（"陕历博""兵马俑"这类专名只有 3-gram 才能整词命中），
    2-gram 用于提升召回。单字中文段落直接整体作为 token。
    """
    q = _strip_filler(q)
    toks = re.findall(r"[A-Za-z0-9]+", q)
    for seg in re.findall(r"[一-龥]+", q):
        if len(seg) == 1:
            toks.append(seg)
            continue
        toks += [seg[i:i + 3] for i in range(len(seg) - 2)]   # 3-gram（长词更具体，权重更高）
        toks += [seg[i:i + 2] for i in range(len(seg) - 1)]   # 2-gram
    return toks


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（滚动数组 DP，O(len(a)×len(b))）

    为什么需要它：中文没有分词器时只能滑窗切 n-gram，而 n-gram 的 IDF 会失效——
    "乾陵门票和开放时间"会切出"开放时""放时间"这类跨词边界的伪词，
    它们因为"全语料只有一条标题这么写"而拿到极高 IDF，
    结果《参考价与时效性提醒》这条通用条目反超真正的《乾陵》条目。
    最长公共子串直接衡量"标题和问题共享多长的连续片段"，
    不受跨词边界伪词影响：乾陵↔乾陵 得 2，通用条目↔问题只有 1，排序立刻正确。
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _title_heads(title: str) -> list[str]:
    """标题里的「用户会直接拿来提问的名字」候选

    《陕西历史博物馆（陕历博）》→ ["陕西历史博物馆", "陕历博"]
    《乾陵（唐高宗与武则天合葬陵）》→ ["乾陵", "唐高宗", "武则天合葬陵"]
    《大雁塔与大唐不夜城》→ ["大雁塔", "大唐不夜城"]

    用途只有一个：判断「问题是不是拿这个名字开的头」。
    这是全语料里最强的召回信号——用户问"乾陵门票和开放时间"，
    就是要《乾陵》这一条，而不是标题里恰好也含"乾陵"的《乾陵陪葬墓与乾陵博物馆》。
    两者在 LCS 与 IDF 上完全同分，只有「谁的名字在问题开头」能把它们分开。
    """
    return [p for p in re.split(r"[（）()、，,·与和及/｜|]", title) if p]


def _score(query: str, doc: str, title: str, dist: float) -> float:
    """混合打分：标题被问题引用 ≫ 标题最长公共子串（平方加权）> IDF 加权标题词命中 > 正文词命中 > 向量

    每个设计点都是实测踩坑之后定的：

    ① 关键词是主信号，向量只当同分兜底。
       ChromaDB 默认 embedding 是英文模型 all-MiniLM-L6-v2，对中文语义区分度极差——
       实测纯向量 top10 里经常根本没有正确条目（"何尊""乾陵""杜虎符"全中招），
       重排再准也救不回来。语料在百条量级，直接全量打分，召回有保证、耗时约 0.2s。

    ② 标题要按"共享连续片段长度"打分，而不是数命中了几个 n-gram。
       见 _lcs_len 的说明：n-gram 计数会被跨词伪词带偏。

    ③ 正文命中按 IDF 加权、权重压到标题的 1/2 以下。
       正文里"门票""预约"这类词到处都是，只有当它真的稀有时才该有分量。

    语料里不存在的 token 直接跳过：既不可能命中，也不该拿默认权重干扰排序。
    """
    idf = _idf()
    head = body = 0.0
    for t in set(_tokens(query)):
        w = idf.get(t)
        if w is None:
            continue
        if t in title:
            head += w * (2.5 if len(t) >= 3 else 1.5)
        if t in doc:
            body += w * (1.5 if len(t) >= 3 else 1.0)
    span = _lcs_len(_strip_filler(query), _strip_filler(title))
    # ④ 问题以某条目的名字开头 → 强信号。
    #    名字越长越具体（"陕西历史博物馆秦汉馆" 压过 "陕西历史博物馆"），
    #    所以按 len² 加权，避免短名字（"乾陵"）把更专的条目（"乾陵陪葬墓"）顶掉。
    q = _strip_filler(query)
    named = max((len(h) ** 2 * 3 for h in _title_heads(_strip_filler(title))
                 if q.startswith(h)), default=0.0)
    return span * span * 5 + named + head * 2 + body + 0.5 * max(0.0, 1.0 - dist), span, named


def _idf() -> dict[str, float]:
    """语料级 IDF（缓存）：token 越稀有，区分度越高"""
    global _idf_cache
    if _idf_cache is None:
        docs = _all_docs(_collection())
        df: dict[str, int] = {}
        for _, doc, meta in docs:
            for t in set(_tokens(doc)) | set(_tokens(meta["title"])):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(docs))
        _idf_cache = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    return _idf_cache


def _all_docs(col) -> list[tuple[str, str, dict]]:
    """全量语料（几十到几百条量级很小）：一次性拉进内存并缓存

    检索改为在全量上打分后，候选不再是"向量 top10"，而是整库——
    这直接解决了"目标条目没被召回就永远找不回来"的问题。
    """
    global _docs
    if _docs is None:
        got = col.get(include=["documents", "metadatas"])
        _docs = list(zip(got["ids"], got["documents"], got["metadatas"]))
    return _docs


@mcp.tool()
async def search_knowledge(query: str) -> str:
    """检索陕西文旅知识库（景点背景、文物故事、参观攻略、行程方法论、美食贴士）。任何涉及景点介绍、文物历史、游玩攻略、行程规划建议的问题，必须先调用本工具获取知识再回答，禁止凭模型记忆作答。

    Args:
        query: 检索词，如 "何尊 中国一词"、"兵马俑 参观攻略"、"华山 路线"
    """
    # 注意：必须是 async def —— FastMCP 会把 sync 工具丢进 anyio 工作线程执行，
    # 而 onnxruntime（ChromaDB 内置 embedding）在非主线程初始化会直接段错误崩进程（实测踩坑）。
    # async 工具在 Server 事件循环（主线程）执行，查询耗时约 0.1-0.3s，可接受。
    try:
        col = _collection()
        if col.count() == 0:
            return "知识库暂时为空，请基于通用知识回答，并提示用户答案未经知识库校验。"
        # 向量距离：全量取一遍，只用于同分兜底（英文 embedding 对中文区分度差，不作主信号）
        res = col.query(query_texts=[query], n_results=col.count())
        dists = dict(zip(res["ids"][0], res["distances"][0]))
        scored = [(*_score(query, doc, meta["title"], dists.get(cid, 1.0)), doc, meta)
                  for cid, doc, meta in _all_docs(col)]
        # 相关度下限：没有一条标题与问题共享 2 个连续汉字（也没被问题点名）→ 判定未收录。
        # 为什么必须有：打分函数永远会返回一个"最像的"，哪怕语料里根本没有这条。
        # 实测问"黄帝陵要不要去"（库内确实没有），首位返回的是《Biangbiang 面与陕西面条谱系》——
        # 模型拿到这种垃圾上下文，比拿到"未收录"更容易编出一个像样的答案。
        hits = [s for s in scored if s[1] >= MIN_SPAN or s[2] > 0]
        if not hits:
            return ("知识库未收录与该问题相关的条目。请明确告知用户：本知识库不包含此内容，"
                    "不要凭记忆编造年代、票价、开放时间等具体事实，可建议其查官方渠道。")
        hits.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, _, _, doc, meta in hits[:TOP_K]:
            out.append(f"【{meta['title']}】（来源：{meta['source']}）\n{doc}")
        return ("\n\n---\n\n".join(out))[:MAX_CHARS]
    except Exception as e:
        # RAG 是增强不是依赖：检索挂了返回提示语，Agent 主流程继续
        # 带上异常类型：本地检索异常若 str(e) 为空，只有类型名能定位
        return f"知识库检索异常（{type(e).__name__}：{str(e).strip() or '无详情'}），请基于通用知识谨慎回答。"


@mcp.tool()
async def rebuild_knowledge() -> str:
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
        _collection()  # 启动时在主线程预加载 embedding 模型（避免首个请求才初始化）
        mcp.run(transport="stdio")
