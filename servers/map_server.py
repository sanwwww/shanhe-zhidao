# -*- coding: utf-8 -*-
"""
山河智导 · 地图路线 MCP Server（高德开放平台）
- 需要环境变量 AMAP_KEY（高德个人开发者免费申请：https://lbs.amap.com）
- 无 Key 时优雅降级：返回引导信息而不是报错崩溃（沿用"增强组件失败不传导主流程"的设计）
启动：python servers/map_server.py
"""
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("amap")
BASE = "https://restapi.amap.com"
NO_KEY_MSG = (
    "地图服务未配置高德 Key（环境变量 AMAP_KEY）。"
    "可到高德开放平台 lbs.amap.com 免费申请个人开发者 Key 后填入 .env。"
    "临时建议：让用户直接使用高德地图 App 查询路线。"
)


def _key() -> str | None:
    return os.getenv("AMAP_KEY") or None


async def _geocode(place: str, city: str = "") -> tuple[float, float, str] | None:
    """地名 → (经度, 纬度, 规范化地名)"""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/v3/geocode/geo",
                        params={"address": place, "city": city, "key": _key()})
        d = r.json()
        geos = d.get("geocodes") or []
        if not geos:
            return None
        lng, lat = geos[0]["location"].split(",")
        return float(lng), float(lat), geos[0].get("formatted_address") or place


@mcp.tool()
async def plan_route(origin: str, destination: str, mode: str = "transit", city: str = "西安") -> str:
    """规划两个地点之间的出行路线。用户问"怎么去/怎么走/从A到B/路线/交通"时使用。

    Args:
        origin: 出发地，如 "西安北站"、"钟楼"
        destination: 目的地，如 "秦始皇兵马俑博物馆"、"宝鸡青铜器博物院"
        mode: 出行方式：transit(公交地铁，默认) / driving(驾车) / walking(步行，3公里内合适)
        city: 出发地所在城市（用于消除同名地名歧义，如"钟楼"全国有多处），默认 "西安"；跨城出行时填出发城市
    """
    if not _key():
        return NO_KEY_MSG
    try:
        o = await _geocode(origin, city)  # 出发地带城市约束，防止"钟楼"匹配到常州等同名地
        t = await _geocode(destination)   # 目的地不约束，支持跨城（如西安→宝鸡）
        if not o:
            return f"未找到出发地「{origin}」，请换更精确的写法（如加城市名）"
        if not t:
            return f"未找到目的地「{destination}」，请换更精确的写法（如加城市名）"
        olnglat, tlnglat = f"{o[0]},{o[1]}", f"{t[0]},{t[1]}"

        async with httpx.AsyncClient(timeout=10) as c:
            if mode == "walking":
                r = await c.get(f"{BASE}/v3/direction/walking",
                                params={"origin": olnglat, "destination": tlnglat, "key": _key()})
                route = r.json().get("route", {})
                paths = route.get("paths") or []
                if not paths:
                    return "未规划出步行路线（距离可能过远，建议换公交/驾车）"
                p = paths[0]
                dist, dur = int(p["distance"]), int(p["duration"])
                steps = " → ".join(s["instruction"] for s in p.get("steps", [])[:6])
                return (f"🚶 步行路线：{o[2]} → {t[2]}\n"
                        f"距离约 {dist/1000:.1f} 公里，预计 {dur//60} 分钟\n参考路径：{steps}")
            if mode == "driving":
                r = await c.get(f"{BASE}/v3/direction/driving",
                                params={"origin": olnglat, "destination": tlnglat, "key": _key()})
                paths = r.json().get("route", {}).get("paths") or []
                if not paths:
                    return "未规划出驾车路线"
                p = paths[0]
                dist, dur = int(p["distance"]), int(p["duration"])
                return (f"🚗 驾车路线：{o[2]} → {t[2]}\n"
                        f"距离约 {dist/1000:.1f} 公里，预计 {dur//60} 分钟，过路费约 {p.get('tolls', '0')} 元")
            # transit 公交
            r = await c.get(f"{BASE}/v3/direction/transit/integrated",
                            params={"origin": olnglat, "destination": tlnglat,
                                    "city": city, "cityd": city, "key": _key()})
            transits = r.json().get("route", {}).get("transits") or []
            if not transits:
                return "未规划出公交路线（可尝试 mode=driving 查询驾车方案）"
            t0 = transits[0]
            dur = int(t0.get("duration", 0))
            segs = []
            for seg in t0.get("segments", [])[:5]:
                bus = (seg.get("bus", {}).get("buslines") or [{}])[0].get("name")
                if bus:
                    segs.append(bus)
                elif seg.get("walking"):
                    segs.append("步行")
            return (f"🚌 公交方案：{o[2]} → {t[2]}\n"
                    f"全程约 {dur//60} 分钟，票价约 {t0.get('cost', '?')} 元\n"
                    f"换乘路径：{' → '.join(segs)}")
    except Exception as e:
        return f"路线查询失败：{e}。建议直接使用高德地图 App 查询。"


@mcp.tool()
async def search_place(keyword: str, city: str = "西安") -> str:
    """搜索某城市内的景点/餐饮/酒店等场所位置。用户问"附近有什么/XX在哪里/推荐住宿餐饮"时使用。

    Args:
        keyword: 搜索关键词，如 "兵马俑"、"肉夹馍"、"钟鼓楼附近酒店"
        city: 城市名，默认 "西安"
    """
    if not _key():
        return NO_KEY_MSG
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{BASE}/v3/place/text",
                            params={"keywords": keyword, "city": city,
                                    "citylimit": "true", "output": "json", "key": _key()})
            pois = r.json().get("pois") or []
        if not pois:
            return f"在{city}未找到「{keyword}」相关场所"
        lines = [f"📍 {city}·{keyword} 搜索结果（前{min(5, len(pois))}个）："]
        for p in pois[:5]:
            lines.append(f"  · {p['name']}｜{p.get('type', '').split(';')[0]}｜{p.get('address', '地址不详')}")
        return "\n".join(lines)
    except Exception as e:
        return f"地点搜索失败：{e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
