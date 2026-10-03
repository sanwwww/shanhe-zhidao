# -*- coding: utf-8 -*-
"""
山河智导 · 文旅知识库 MCP Server（RAG-as-MCP-Tool）
- 把 RAG 检索封装成一个标准 MCP 工具：对 Agent 来说，知识库和天气、地图是同构的工具
- 检索策略：全量打分（标题被引用 ≫ 标题最长公共子串 > IDF 加权词命中），取 top3
- 语料直接在进程内解析，**不引入任何向量库**。这不是简化，是止血，见下。

为什么这里没有 ChromaDB / embedding
-----------------------------------
原本用 ChromaDB 存语料 + 默认英文 embedding（all-MiniLM-L6-v2）做"向量同分兜底"。
2026-10-03 线上验收抓到：**任何一次 search_knowledge 都会把整个服务打死**——
SSE 流既不给 done 也不给 error（说明不是 Python 异常，是进程被 SIGKILL），
首次检索 16.5 秒后连接断掉，之后 45 次请求全部 502，直到 Render 拉起新实例。

本机实测内存（psutil，单进程）：

    裸解释器                                    21 MB
    import knowledge_server                     62 MB
    建 Chroma 集合（含 ONNX embedding 初始化）   107 MB
    首次检索完成                               138 MB  ← 峰值 271 MB

知识库是独立 MCP 子进程，它 peak 271MB，再叠加主进程（FastAPI+LangGraph）
与另外 3 个 MCP 子进程，超过 Render 免费版 512MB 上限 → OOM kill。

代价与收益完全不成比例：向量项在 _score 里权重只有 0.5，
而标题信号 span²*5 最高可到 45+；语料本身在百条量级，早已全量驻留内存。
也就是说，为了一个"同分兜底"项，付出的是整个服务在生产环境被随机制杀。

去掉之后：无 onnxruntime（顺带消除它在非主线程初始化会段错误崩进程的隐患）、
无磁盘索引、无懒加载、冷启动更快。检索质量由 tests/test_core.py 的
首位命中用例守住——不靠"应该有影响不大"的口头保证。

- 空库自愈：语料直接来自 knowledge/*.md，部署到新容器天然就是最新内容，
  不再需要"容器磁盘重置 → 检测空库 → 重新入库"这条保险丝
启动：python servers/knowledge_server.py
"""
import math
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
MAX_CHARS = 2000   # 返回给模型的检索片段上限：知识库扩到 90+ 条后单条最长约 370 字，
                   # 3 条拼接最大约 1100 字；留到 2000 是为了不截断
TOP_K = 3          # 最终返回给模型的条目数（打分已按"标题命中 ≫ 正文命中"排序）
MIN_SPAN = 2       # 相关度下限：标题与问题至少要共享 2 个连续汉字，才认为"问到了这个条目"

mcp = FastMCP("knowledge")

_docs = None       # 全量语料内存缓存（检索在全量上打分，见 _all_docs）
_idf_cache = None  # 语料级 IDF 缓存（打分用，见 _idf）


def _parse_corpus() -> list[tuple[str, str, dict]]:
    """把 knowledge/*.md 解析成 (id, 文档块, 元数据) 列表

    按 '## ' 二级标题切块：一个标题=一个完整知识点，语义不断裂。
    切不出任何块就直接抛——空语料下检索会"诚实地"回未收录，
    但那种正常态和故障态无法区分，宁可启动失败。
    """
    docs: list[tuple[str, str, dict]] = []
    for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, chunk in enumerate(re.split(r"(?m)^(?=## )", text)):
            chunk = chunk.strip()
            if len(chunk) < 30 or not chunk.startswith("## "):
                continue
            title = chunk.splitlines()[0].lstrip("# ").strip()
            docs.append((f"{md.stem}-{i}", chunk, {"source": md.name, "title": title}))
    if not docs:
        raise RuntimeError(f"知识库为空或格式不符：{KNOWLEDGE_DIR} 未解析出任何条目")
    return docs


def _reload() -> int:
    """重新解析语料并使缓存失效（等价于原来的"重建索引"）"""
    global _docs, _idf_cache
    _docs = _parse_corpus()
    _idf_cache = None
    return len(_docs)


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


def _score(query: str, doc: str, title: str) -> tuple[float, int, float]:
    """混合打分：标题被问题引用 ≫ 标题最长公共子串（平方加权）> IDF 加权标题词命中 > 正文词命中

    每个设计点都是实测踩坑之后定的：

    ① 关键词是主信号。语料在百条量级，直接全量打分，召回有保证、耗时约 0.2s。
       曾经用向量相似度做同分兜底，现已移除——原因见模块文档（它把生产环境打死了），
       而且权重只有 0.5，对排序基本无影响。

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
    return span * span * 5 + named + head * 2 + body, span, named


def _idf() -> dict[str, float]:
    """语料级 IDF（缓存）：token 越稀有，区分度越高"""
    global _idf_cache
    if _idf_cache is None:
        docs = _all_docs()
        df: dict[str, int] = {}
        for _, doc, meta in docs:
            for t in set(_tokens(doc)) | set(_tokens(meta["title"])):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(docs))
        _idf_cache = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    return _idf_cache


def _all_docs() -> list[tuple[str, str, dict]]:
    """全量语料：进程内一次性解析并缓存（百条量级，内存开销可忽略）

    检索在全量上打分，候选是整库——这直接解决了"目标条目没被召回就永远找不回来"的问题。
    """
    global _docs
    if _docs is None:
        _docs = _parse_corpus()
    return _docs


@mcp.tool()
async def search_knowledge(query: str) -> str:
    """检索陕西文旅知识库（景点背景、文物故事、参观攻略、行程方法论、美食贴士）。任何涉及景点介绍、文物历史、游玩攻略、行程规划建议的问题，必须先调用本工具获取知识再回答，禁止凭模型记忆作答。

    Args:
        query: 检索词，如 "何尊 中国一词"、"兵马俑 参观攻略"、"华山 路线"
    """
    # 仍是 async：FastMCP 对 sync 工具会丢进 anyio 工作线程执行，async 工具则留在
    # Server 事件循环（主线程）。检索本身只有约 0.06s，留在主线程既省线程又不改语义。
    # 历史上这里是为了绕开 onnxruntime 在工作线程初始化会段错误崩进程；
    # 现在 onnxruntime 已经移除，但保持 async 的调用契约不变（smoke_test 与守卫测试都按 async 校验）。
    try:
        docs = _all_docs()
        scored = [(*_score(query, doc, meta["title"]), doc, meta) for _, doc, meta in docs]
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
        return f"知识库已重建，共 {_reload()} 条知识。"
    except Exception as e:
        return f"重建失败：{type(e).__name__}：{str(e).strip() or '无详情'}"


if __name__ == "__main__":
    if "--build" in sys.argv:
        # 保留这个入口只是为了不改动 Render 的 buildCommand（改 render.yaml 需要在
        # 控制台重新同步 Blueprint，没必要为一行命令增加部署风险）。
        # 现在它做的事：解析并校验语料可读、条目非空。
        print(f"知识库就绪，共 {len(_parse_corpus())} 条")
    else:
        mcp.run(transport="stdio")
