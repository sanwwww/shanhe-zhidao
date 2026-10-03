# -*- coding: utf-8 -*-
"""单元测试（全部不调 LLM，零成本）：python -m pytest tests/ -v"""
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from servers.guide_server import generate_guide, list_guide_spots          # noqa: E402
from servers.knowledge_server import search_knowledge, _collection          # noqa: E402
from servers.weather_server import get_clothing_advice                      # noqa: E402
from agent_core import load_servers_config                                  # noqa: E402


# ── 知识库 RAG ─────────────────────────────
def test_kb_built():
    assert _collection().count() >= 80, "知识库条目数应 ≥80（覆盖西安/咸阳/宝鸡/延安/陕南陕北）"


def test_kb_hezun_hit():
    """"何尊/宅兹中国" 查询必须命中何尊条目（混合检索质量红线）"""
    r = asyncio.run(search_knowledge("何尊 中国一词最早的记载"))
    assert "何尊" in r and "宅兹中国" in r


def test_kb_bingmayong_hit():
    r = asyncio.run(search_knowledge("兵马俑参观攻略门票"))
    assert "兵马俑" in r


# 检索质量回归：这些 query 覆盖"英文 embedding 中文召回失败"与
# "n-gram 伪词抢走首位"两类高危场景（专名、地名、文物名、填充词干扰）。
# 旧实现（向量 topN → 关键词等权重排）实测在"何尊""乾陵""杜虎符""壶口""华山夜爬"
# 上会召回错误条目；改为「全量打分 + IDF + 标题最长公共子串」后逐条钉死，防止改回去。
RETRIEVAL_CASES = [
    ("何尊为什么重要", "何尊"),
    ("乾陵门票和开放时间", "乾陵"),
    ("杜虎符是用来做什么的", "杜虎符"),
    ("壶口瀑布什么时候去最好", "壶口"),
    ("太白山天圆地方", "太白山"),
    ("延安宝塔山门票", "宝塔山"),
    ("汉中油菜花什么时候开", "油菜花"),
    ("陕历博怎么预约", "陕西历史博物馆"),
    ("铜川耀州窑陈炉古镇", "耀州窑"),
    ("法门寺佛指舍利", "法门寺"),
    ("金丝峡避暑", "金丝峡"),
    ("葡萄花鸟纹银香囊的原理", "香囊"),
    ("西安城墙骑行", "城墙"),
    ("给小学生怎么讲兵马俑", "兵马俑"),
    # 填充词干扰：这两条旧实现会被《回民街、洒金桥与永兴坊怎么选》用"怎么"抢走首位
    ("华山夜爬怎么安排", "华山"),
    ("大雁塔值得上去吗", "大雁塔"),
]


@pytest.mark.parametrize("query,expect_top1", RETRIEVAL_CASES)
def test_kb_retrieval_top1(query, expect_top1):
    """首位命中：正确条目必须排在检索结果第一条（而非只是"在结果里"）"""
    r = asyncio.run(search_knowledge(query))
    first = next((l for l in r.splitlines() if l.startswith("【")), "")
    assert expect_top1 in first, f"query={query!r} 首位应为含「{expect_top1}」的条目，实际首位：{first}"


@pytest.mark.parametrize("query,expect_anywhere", [
    # 这类问题没有唯一正确条目，只要求正确内容出现在返回的 top3 内
    ("咸阳帝陵一日游怎么安排", "乾陵"),
    ("延安有哪些必去的红色景点", "宝塔山"),
    ("宝鸡青铜器博物院有什么镇馆之宝", "何尊"),
    ("西安三天怎么安排最经典", "行程|D1|第一天"),
])
def test_kb_retrieval_top3(query, expect_anywhere):
    r = asyncio.run(search_knowledge(query))
    assert re.search(expect_anywhere, r), f"query={query!r} 结果中应包含「{expect_anywhere}」，实际：{r[:300]}"


def test_strip_filler_removes_low_information_words():
    """填充词必须被剥离，否则"怎么""什么"会造出假的长公共子串"""
    from servers.knowledge_server import _strip_filler
    assert _strip_filler("华山夜爬怎么安排") == "华山夜爬"
    assert _strip_filler("汉中油菜花什么时候开") == "汉中油菜花开"
    assert _strip_filler("何尊为什么重要") == "何尊重要"


# ── 相关度下限：库里没有的，必须承认没有 ─────────────────────────────
# 打分函数永远会返回一个"最像的"，哪怕语料里根本没有这条。
# 实测问"黄帝陵要不要去"（当时库里确实没有），首位返回的是《Biangbiang 面与陕西面条谱系》——
# 模型拿到这种垃圾上下文，比拿到"未收录"更容易编出一个像样的答案。
NO_MATCH_CASES = [
    "三亚有什么好玩的",
    "库里面没有的地方",
    "日本京都怎么玩",
    "上海迪士尼门票多少",
]


@pytest.mark.parametrize("query", NO_MATCH_CASES)
def test_kb_out_of_corpus_reports_not_covered(query):
    """库外提问必须返回「未收录」，而不是硬凑一个最像的条目"""
    r = asyncio.run(search_knowledge(query))
    assert "未收录" in r, f"query={query!r} 应判定未收录，实际返回：{r[:200]}"


def test_kb_relevance_floor_uses_span_not_body_noise():
    """下限判的是"标题是否被问题引用"，不是"正文里凑巧有同义词"。

    "杭州西湖怎么去"与《乾陵》正文都可能出现"帝陵/陵"之类的字，
    这类正文噪声不该让一个完全无关的条目通过下限。
    """
    from servers.knowledge_server import MIN_SPAN
    assert MIN_SPAN >= 2, "下限低于 2 个连续汉字会放进大量噪声"
    r = asyncio.run(search_knowledge("杭州西湖怎么去"))
    assert "未收录" in r


# ── 标题别名：用户会用的问法要写进标题 ──────────────────────────────
def test_title_heads_extracts_aliases():
    """标题里的正名、括号别名、并列项都要被当成"用户会用的名字" """
    from servers.knowledge_server import _title_heads
    assert "陕历博" in _title_heads("陕西历史博物馆（陕历博）")
    assert "陕西历史博物馆" in _title_heads("陕西历史博物馆（陕历博）")
    assert "乾陵" in _title_heads("乾陵（唐高宗与武则天合葬陵）")
    # 并列标题要拆开，否则"大雁塔值得上去吗"匹配不上《大雁塔与大唐不夜城》
    assert set(_title_heads("大雁塔与大唐不夜城")) == {"大雁塔", "大唐不夜城"}


@pytest.mark.parametrize("query,expect_top1", [
    # 用户在问题里用的是"行程说法"，不是条目的正名——别名进标题才能命中
    ("咸阳帝陵一日游怎么安排", "行程规划"),
    ("西安三天怎么玩", "行程规划"),
    ("黄帝陵门票多少", "黄帝陵"),
])
def test_kb_alias_query_hits_right_entry(query, expect_top1):
    r = asyncio.run(search_knowledge(query))
    first = next((l for l in r.splitlines() if l.startswith("【")), "")
    assert expect_top1 in first, f"query={query!r} 首位应为含「{expect_top1}」的条目，实际：{first}"


def test_kb_has_no_empty_or_duplicate_titles():
    """标题是检索的主键：空标题或重名会让打分失去意义"""
    from servers.knowledge_server import _collection, _all_docs
    titles = [m["title"].strip() for _, _, m in _all_docs(_collection())]
    assert all(titles), "存在空标题"
    assert len(titles) == len(set(titles)), f"标题重复：{len(titles)} 条中有重名"


def test_kb_no_stale_chunks_after_rebuild():
    """重建必须清空旧块：语料改标题后，旧 id 不能以孤儿身份留在库里"""
    col = _collection()
    ids = col.get(include=[])["ids"]
    assert len(ids) == len(set(ids))
    for cid in ids:
        stem = cid.rsplit("-", 1)[0]
        assert (Path(__file__).resolve().parent.parent / "knowledge" / f"{stem}.md").exists(), \
            f"库里存在来源已删除的孤儿块：{cid}"


def test_kb_graceful_on_garbage():
    """乱 query 也不能抛异常（RAG 是增强不是依赖）"""
    r = asyncio.run(search_knowledge("asdfghjkl12345"))
    assert isinstance(r, str) and len(r) > 0


# ── 讲解词 ────────────────────────────────
def test_guide_audience():
    r = generate_guide("何尊", "儿童")
    assert "儿童" in r and "讲解词要点" in r


def test_guide_unknown_spot():
    r = generate_guide("不存在的景点xyz")
    assert "暂未收录" in r


def test_guide_list_groups_by_source():
    """已收录素材从几条扩到 90+ 后，列表必须分组并给出总数（否则输出会把上下文挤爆）"""
    r = list_guide_spots()
    assert "共" in r and "条" in r
    assert "何尊" in r and "【" in r


def test_guide_unknown_lists_scope():
    r = generate_guide("不存在的景点xyz")
    assert "暂未收录" in r
    assert "可讲范围" in r


# ── 天气（纯函数部分）───────────────────────
def test_clothing():
    assert "外套" in get_clothing_advice("秋")
    assert "请说明季节" in get_clothing_advice("火星季")


# ── MCP 配置 ──────────────────────────────
def test_servers_config_absolute_and_safe():
    """Server 路径必须解析为绝对路径，command 必须锁定当前解释器（防 PATH 上裸 python 无依赖）"""
    cfg = load_servers_config()
    assert len(cfg) == 4
    for s in cfg.values():
        assert s["command"] == sys.executable
        assert Path(s["args"][0]).is_absolute()
        assert Path(s["args"][0]).exists()


def test_testset_valid():
    """测试集结构校验：判据完整性由 tests/test_scoring.py 深度覆盖，这里只做粗校验"""
    items = json.loads((ROOT / "evaluation" / "testset.json").read_text(encoding="utf-8"))
    assert len(items) >= 45
    assert len({i["id"] for i in items}) == len(items)
    for it in items:
        assert (it.get("expected_tools") or it.get("must_match")
                or it.get("must_not_match") or it.get("keypoints"))


# ── 冒烟脚本与工具签名一致性 ─────────────────────────────
def test_smoke_test_awaits_every_async_tool():
    """冒烟脚本调用 async 工具函数时必须 await。

    背景：`search_knowledge` 为了绕开 FastMCP 工作线程里的 onnxruntime 段错误
    改成了 async，但 smoke_test.py 没跟着改，于是 `print(结果[:400])` 变成
    `TypeError: 'coroutine' object is not subscriptable`——冒烟脚本从那天起就是坏的，
    而 README 还把 `python smoke_test.py` 列为质量保障手段。静默失效最危险，
    所以这里用 AST 静态钉死，不依赖有没有人记得手跑。
    """
    import ast
    import importlib
    import inspect

    async_names = set()
    for mod_name in ("servers.weather_server", "servers.guide_server",
                     "servers.knowledge_server", "servers.map_server"):
        mod = importlib.import_module(mod_name)
        async_names |= {n for n in dir(mod) if inspect.iscoroutinefunction(getattr(mod, n))}

    tree = ast.parse((ROOT / "smoke_test.py").read_text(encoding="utf-8"))
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    offenders = [
        f"第 {node.lineno} 行调用 {node.func.id}() 未 await"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in async_names
        and not isinstance(parents.get(id(node)), ast.Await)
    ]
    assert not offenders, "；".join(offenders)


def test_every_registered_tool_is_reachable():
    """servers_config.json 里声明的 4 个 Server，其脚本必须都存在且能起得来。

    典型翻车：改了文件名忘了改配置，MCP 子进程启动即失败，
    表现是"工具凭空少了一个"——线上很难注意到。
    """
    cfg = load_servers_config()
    assert set(cfg) == {"weather", "amap", "knowledge", "guide"}
    for name, spec in cfg.items():
        script = Path(spec["args"][0])
        assert script.exists(), f"{name} 的入口脚本不存在：{script}"
