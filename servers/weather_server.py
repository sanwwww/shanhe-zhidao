# -*- coding: utf-8 -*-
"""
山河智导 · 天气 MCP Server
- 默认用 Open-Meteo（免费、免 Key），可选切换和风天气（设 QWEATHER_KEY 环境变量）
- 面试点：工具的 docstring 就是给 LLM 看的"说明书"，写清楚什么时候该用它
启动：python servers/weather_server.py （由 MCP Client 以 stdio 拉起）
"""
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

# Open-Meteo WMO 天气码 → 中文（常见项）
WMO_CN = {
    0: "晴", 1: "大致晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "大冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨", 85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


async def _geocode(city: str) -> dict | None:
    """城市名 → 经纬度（Open-Meteo 地理编码，免 Key）"""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        results = r.json().get("results") or []
        return results[0] if results else None


@mcp.tool()
async def get_weather(city: str) -> str:
    """查询指定城市/景区所在地的实时天气和未来3天预报。用户问"天气怎么样/要不要带伞/适合出游吗/穿什么"时使用。

    Args:
        city: 城市名，如 "西安"、"宝鸡"、"华山"
    """
    try:
        if os.getenv("QWEATHER_KEY"):
            return await _qweather(city)
        geo = await _geocode(city)
        if not geo:
            return f"未找到地点「{city}」，请确认地名（可换成所属城市名再试）"
        lat, lon = geo["latitude"], geo["longitude"]
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "Asia/Shanghai", "forecast_days": 3,
                },
            )
            d = r.json()
        cur = d["current"]
        lines = [
            f"📍 {geo.get('name', city)}（{geo.get('admin1', '')}）实时天气",
            f"🌡 气温 {cur['temperature_2m']}°C（体感 {cur['apparent_temperature']}°C），湿度 {cur['relative_humidity_2m']}%，风速 {cur['wind_speed_10m']}km/h",
            f"🌤 天气：{WMO_CN.get(cur['weather_code'], '未知')}",
            "📅 未来3天：",
        ]
        daily = d["daily"]
        for i in range(len(daily["time"])):
            lines.append(
                f"  {daily['time'][i]}：{WMO_CN.get(daily['weather_code'][i], '?')}，"
                f"{daily['temperature_2m_min'][i]}~{daily['temperature_2m_max'][i]}°C，"
                f"降水概率 {daily['precipitation_probability_max'][i]}%"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败：{e}。请稍后重试或换个地名。"


async def _qweather(city: str) -> str:
    """和风天气（需 QWEATHER_KEY，国内更稳）"""
    key = os.environ["QWEATHER_KEY"]
    async with httpx.AsyncClient(timeout=10) as c:
        g = await c.get("https://geoapi.qweather.com/v2/city/lookup",
                        params={"location": city, "key": key})
        locs = g.json().get("location") or []
        if not locs:
            return f"未找到地点「{city}」"
        loc = locs[0]
        w = await c.get("https://devapi.qweather.com/v7/weather/3d",
                        params={"location": loc["id"], "key": key})
        days = w.json().get("daily") or []
    lines = [f"📍 {loc['name']}（{loc['adm1']}）3天预报："]
    for d in days:
        lines.append(
            f"  {d['fxDate']}：白天{d['textDay']}/夜间{d['textNight']}，"
            f"{d['tempMin']}~{d['tempMax']}°C，{d['windDirDay']}{d['windScaleDay']}级"
        )
    return "\n".join(lines)


@mcp.tool()
def get_clothing_advice(season: str) -> str:
    """根据季节给出陕西旅游穿搭与装备建议。用户问"这个天气穿什么/带什么装备"时配合 get_weather 使用。

    Args:
        season: 季节，如 "春"、"夏"、"秋"、"冬"
    """
    tips = {
        "春": "🧥 陕西春季早晚温差大（可达10°C+），建议薄外套+长袖叠穿；春季多风沙，备口罩和墨镜。",
        "夏": "🧢 西安夏季炎热（35°C+常见），速干衣+防晒+遮阳帽；兵马俑馆内闷热，带水杯和小风扇。",
        "秋": "🍂 秋季是陕西最佳旅游季，长袖+薄外套即可；爬华山备防风外套，山顶温度低5-8°C。",
        "冬": "🧣 冬季干冷（-5~8°C），羽绒服+保暖内衣必备；室内有暖气，洋葱式穿搭方便穿脱。",
    }
    return tips.get(season.strip().replace("季", ""), "请说明季节（春/夏/秋/冬）")


if __name__ == "__main__":
    mcp.run(transport="stdio")
