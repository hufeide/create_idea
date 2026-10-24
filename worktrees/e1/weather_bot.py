"""天气查询机器人

用户输入城市名（中文或英文），返回未来 3 天的天气。
数据源：Open-Meteo（免费，无需 API key）。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10

# 常见中文城市名 → 英文地名（Open-Meteo 地名库以英文名为主）
CN_CITY_ALIASES = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "成都": "Chengdu",
    "杭州": "Hangzhou",
    "重庆": "Chongqing",
    "武汉": "Wuhan",
    "西安": "Xi'an",
    "南京": "Nanjing",
    "天津": "Tianjin",
    "苏州": "Suzhou",
    "长沙": "Changsha",
    "郑州": "Zhengzhou",
    "青岛": "Qingdao",
    "大连": "Dalian",
    "昆明": "Kunming",
    "哈尔滨": "Harbin",
    "沈阳": "Shenyang",
    "西安": "Xi'an",
    "拉萨": "Lhasa",
    "乌鲁木齐": "Urumqi",
    "厦门": "Xiamen",
    "昆明": "Kunming",
    "福州": "Fuzhou",
    "宁波": "Ningbo",
    "无锡": "Wuxi",
    "济南": "Jinan",
    "青岛": "Qingdao",
    "太原": "Taiyuan",
    "银川": "Yinchuan",
    "西宁": "Xining",
    "南宁": "Nanning",
    "海口": "Haikou",
    "三亚": "Sanya",
    "珠海": "Zhuhai",
    "常州": "Changzhou",
    "南通": "Nantong",
    "温州": "Wenzhou",
    "泉州": "Quanzhou",
    "湖州": "Huzhou",
    "绍兴": "Shaoxing",
    "台州": "Taizhou",
    "金华": "Jinhua",
    "嘉兴": "Jiaxing",
    "无锡": "Wuxi",
}

# WMO 天气代码 → 中文描述
WMO_CODES = {
    0: "晴",
    1: "晴",
    2: "少云",
    3: "多云",
    4: "阴天",
    5: "雨",
    6: "雨",
    7: "雪",
    8: "雪",
    9: "雪",
    10: "小阵雨",
    11: "阵雨",
    12: "阵雨",
    21: "小阵雨",
    22: "阵雨",
    23: "阵雨",
    24: "小阵雨",
    25: "小阵雨",
    26: "小阵雨",
    27: "雷阵雨",
    28: "雷阵雨",
    29: "雷阵雨",
    30: "小阵雨",
    31: "阵雨",
    32: "阵雨",
    33: "小阵雨",
    34: "阵雨",
    35: "小阵雨",
    36: "小阵雨",
    37: "小阵雨",
    38: "小阵雨",
    39: "小阵雨",
    45: "雾",
    48: "雾",
    51: "小阵雨",
    53: "阵雨",
    55: "阵雨",
    56: "小阵雨",
    57: "小阵雨",
    61: "小阵雨",
    63: "阵雨",
    65: "阵雨",
    66: "小阵雨",
    67: "小阵雨",
    71: "小阵雨",
    73: "阵雨",
    75: "阵雨",
    77: "小阵雨",
    80: "小阵雨",
    81: "阵雨",
    82: "阵雨",
    85: "小阵雨",
    86: "小阵雨",
    95: "雷阵雨",
    96: "雷阵雨",
    99: "雷阵雨",
}


def _http_json(url: str) -> dict:
    """发送 GET 请求并解析 JSON，失败抛出异常。"""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def geocode(city: str) -> dict:
    """把城市名解析为坐标信息。"""
    name = CN_CITY_ALIASES.get(city.strip(), city.strip())
    params = urllib.parse.urlencode({"name": name, "count": 1, "language": "en"})
    data = _http_json(f"{GEOCODE_URL}?{params}")
    results = data.get("results") or []
    if not results:
        raise ValueError(f"未找到城市: {city}")
    return results[0]


def fetch_forecast(loc: dict, days: int = 3) -> dict:
    """获取坐标点未来 days 天的逐日天气预报。"""
    params = urllib.parse.urlencode({
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "forecast_days": days,
        "timezone": "auto",
    })
    return _http_json(f"{FORECAST_URL}?{params}")


def get_forecast(city: str, days: int = 3) -> dict:
    """按城市名查询未来 days 天天气，返回结构化 dict。"""
    loc = geocode(city)
    data = fetch_forecast(loc, days)
    daily = data.get("daily") or {}
    rows = []
    for i, date in enumerate(daily.get("time") or []):
        rows.append({
            "date": date,
            "weather": WMO_CODES.get(daily.get("weather_code", [None])[i], "未知"),
            "max_temp_c": daily.get("temperature_2m_max", [None])[i],
            "min_temp_c": daily.get("temperature_2m_min", [None])[i],
            "precip_mm": daily.get("precipitation_sum", [None])[i],
            "precip_prob_pct": daily.get("precipitation_probability_max", [None])[i],
            "max_wind_kmh": daily.get("wind_speed_10m_max", [None])[i],
        })
    return {
        "city": loc["name"],
        "country": loc.get("country"),
        "timezone": data.get("timezone"),
        "days": rows,
    }


def format_forecast(forecast: dict) -> str:
    """把结构化天气数据格式化为人类可读文本。"""
    lines = [f"🌤  {forecast['city']}（{forecast.get('country', '')}）未来 3 天天气", "─" * 26]
    for row in forecast["days"]:
        max_t = row["max_temp_c"]
        min_t = row["min_temp_c"]
        wind = row["max_wind_kmh"]
        parts = [row["weather"], f"最高 {max_t}°C / 最低 {min_t}°C"]
        if row["precip_prob_pct"] is not None:
            parts.append(f"降水概率 {row['precip_prob_pct']}%")
        if wind is not None:
            parts.append(f"最大风速 {round(wind, 1)} km/h")
        lines.append(f"  {row['date']}  " + "  |  ".join(parts))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: python weather_bot.py <城市名>   （例如: python weather_bot.py 北京）")
        print("交互模式: python weather_bot.py --interactive")
        return 1

    if argv[1] == "--interactive":
        return _repl()

    try:
        forecast = get_forecast(argv[1])
    except ValueError as e:
        print(str(e))
        return 1
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        print(f"请求天气服务失败: {e}")
        return 1

    print(format_forecast(forecast))
    return 0


def _repl() -> int:
    """交互式模式：持续询问城市名，输入 q 退出。"""
    print("🌤 天气查询机器人（输入城市名查询未来 3 天天气，输入 q 退出）")
    while True:
        try:
            city = input("请输入城市名: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not city:
            continue
        if city.lower() in ("q", "quit", "exit", "退出"):
            break
        try:
            forecast = get_forecast(city)
        except ValueError as e:
            print(str(e))
            continue
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            print(f"请求天气服务失败: {e}")
            continue
        print(format_forecast(forecast))
    print("再见 👋")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
