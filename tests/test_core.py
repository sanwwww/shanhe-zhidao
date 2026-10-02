# -*- coding: utf-8 -*-
"""单元测试（全部不调 LLM，零成本）：python -m pytest tests/ -v"""
import asyncio
import json
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
    assert _collection().count() >= 15, "知识库条目数应 ≥15"


def test_kb_hezun_hit():
    """"何尊/宅兹中国" 查询必须命中何尊条目（混合检索质量红线）"""
    r = asyncio.run(search_knowledge("何尊 中国一词最早的记载"))
    assert "何尊" in r and "宅兹中国" in r


def test_kb_bingmayong_hit():
    r = asyncio.run(search_knowledge("兵马俑参观攻略门票"))
    assert "兵马俑" in r


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


def test_guide_list():
    assert "何尊" in list_guide_spots()


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
    items = json.loads((ROOT / "evaluation" / "testset.json").read_text(encoding="utf-8"))
    assert len(items) == 30
    for it in items:
        assert it["expected_tools"] and it["keypoints"]
