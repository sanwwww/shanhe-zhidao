# -*- coding: utf-8 -*-
"""流式答案过滤 + 工具返回值清洗的回归测试（不调 LLM，零成本）

背景：线上实测抓到两个必现的展示缺陷，本文件用于防止后续改动把修复冲掉。
  1. 模型在发起 tool_call 前会先输出一段英文"前言"，被当成正式回答推给用户
     （实测原句："I'll look this up in the knowledge base." /
                 "I'll look up the knowledge base for details on 大明宫国家遗址公园."）
  2. 工具返回值在 LangChain v1 下是 content-blocks 列表，str() 之后把
     Python 对象 repr 直接推给了前端
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api_server import AnswerStreamFilter, _to_text          # noqa: E402


def run(steps) -> str:
    """按事件顺序驱动 AnswerStreamFilter，返回最终会推给用户的文本。

    steps 元素：("chunk", run_id, text) 表示一次模型流式输出；("tool",) 表示一次工具调用。
    """
    f = AnswerStreamFilter()
    out = []
    for s in steps:
        if s[0] == "chunk":
            f.on_model_chunk(s[1], s[2])
            out.extend(f.drain())
        else:
            f.on_tool_start()
    out.extend(f.finish())
    return "".join(out)


# 线上实测抓到的两句真实前言，长度 39 / 57 字符
PREAMBLE_SHORT = "I'll look this up in the knowledge base."
PREAMBLE_LONG = "I'll look up the knowledge base for details on 大明宫国家遗址公园."


# ── 前言过滤 ─────────────────────────────
def test_short_preamble_dropped():
    """单步工具：调用前的英文前言不能出现在答案里"""
    r = run([("chunk", "A", PREAMBLE_SHORT), ("tool",),
             ("chunk", "B", "# 何尊为什么重要？\n\n因为铭文里有最早的中国二字。")])
    assert "I'll" not in r
    assert "何尊为什么重要" in r


def test_long_preamble_dropped():
    """长前言回归：57 字符，超过任何合理长度阈值，必须靠分段而非长度判断"""
    assert len(PREAMBLE_LONG) > 40          # 确实用例长度足以击穿阈值法
    r = run([("chunk", "A", PREAMBLE_LONG), ("tool",),
             ("chunk", "B", "知识库里目前没有收录大明宫。")])
    assert "I'll" not in r
    assert "大明宫" in r


def test_multi_step_only_last_segment_kept():
    """多步 ReAct：每一轮工具前的前言都要丢，只保留最终答案段"""
    r = run([("chunk", "A", "我先查知识库"), ("tool",),
             ("chunk", "B", "再查一下天气"), ("tool",),
             ("chunk", "C", "以下是完整行程：第一天城墙，第二天兵马俑。")])
    assert "我先查知识库" not in r
    assert "再查一下天气" not in r
    assert "完整行程" in r


def test_no_tool_call_keeps_everything():
    """全程没调工具时，内容必须完整保留，不能把回答也吞掉"""
    r = run([("chunk", "A", "你好，我是山河智导。"),
             ("chunk", "A", "可以帮你规划陕西行程。")])
    assert "你好" in r and "规划陕西行程" in r


def test_answer_segment_split_not_truncated():
    """答案段被拆成多个 chunk 时不能丢字"""
    r = run([("chunk", "A", "查一下"), ("tool",),
             ("chunk", "B", "第一段"), ("chunk", "B", "第二段"), ("chunk", "B", "第三段")])
    assert r == "第一段第二段第三段"


def test_empty_chunk_ignored():
    """空 chunk 不应产生空段或污染输出"""
    r = run([("chunk", "A", ""), ("chunk", "A", "有效内容")])
    assert r == "有效内容"


# ── 工具返回值清洗 ────────────────────────
def test_content_blocks_to_plain_text():
    """线上实测的脏数据形态：[{'type':'text','text':...,'id':...}]"""
    dirty = [{"type": "text", "text": "【何尊】1963年出土于宝鸡", "id": "lc_0e5f9eed"}]
    out = _to_text(dirty)
    assert out == "【何尊】1963年出土于宝鸡"
    assert "lc_" not in out and "'type'" not in out


def test_to_text_various_shapes():
    assert _to_text("纯文本") == "纯文本"
    assert _to_text([{"type": "text", "text": "甲"}, {"type": "text", "text": "乙"}]) == "甲乙"

    class Obj:
        content = [{"type": "text", "text": "嵌套"}]

    assert _to_text(Obj()) == "嵌套"
    assert _to_text(None) == ""
