"""数据获取蛋 · 天气模块

调用天气 API（Open-Meteo，免费且无需 API key），按城市名返回
结构化的当前天气数据（JSON 可序列化 dict）。
"""

from __future__ import annotations

import json
import sys

try:
    import requests
except ImportError:  # pragma: no cover
    raise SystemExit("依赖缺失：请先执行 `pip install requests`")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
}

CURRENT_VARS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,"
    "weather_code,wind_speed_10m,precipitation"
)

# WMO 天气代码 → 中文描述（Open-Meteo 使用）
WEATHER_CODE_NAMES = {
    0: "晴", 1: "晴", 2: "少云", 3: "多云",
    45: "薄雾", 48: "结冰雾",
    51: "毛毛雨", 53: "毛毛雨", 55: "浓毛毛雨",
    56: "冻毛毛雨", 57: "浓冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小冰雹", 86: "大冰雹",
    95: "雷暴", 96: "雷暴伴大冰雹", 99: "雷暴伴大冰雹",
}


def _describe_weather(code: int) -> str:
    """WMO 天气代码转中文天气状况描述。"""
    return WEATHER_CODE_NAMES.get(code, f"未知({code})")


def _geocode(city: str) -> dict:
    """把城市名解析为坐标。"""
    name = CN_CITY_ALIASES.get(city, city)
    resp = requests.get(
        GEOCODE_URL,
        params={"name": name, "count": 1, "language": "en"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"未找到城市: {city}")
    return results[0]


def get_weather(city: str) -> dict:
    """按城市名查询当前天气，返回结构化 dict。

    参数:
        city: 城市名（中文或英文均可）

    返回:
        dict: 包含 city、坐标、温度、体感温度、湿度、风速、
              降水、天气代码与观测时间等字段
    """
    loc = _geocode(city)
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": CURRENT_VARS,
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    cur = resp.json().get("current") or {}

    return {
        "city": loc["name"],
        "country": loc.get("country"),
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "temperature_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "wind_speed_mps": cur.get("wind_speed_10m"),
        "precipitation_mm": cur.get("precipitation"),
        "weather_code": cur.get("weather_code"),
        "observed_at": cur.get("time"),
    }


def get_daily_forecast(city: str, days: int = 3) -> dict:
    """按城市名查询未来 N 天（默认 3 天）的每日天气预报，返回结构化 dict。

    参数:
        city: 城市名（中文或英文均可）
        days: 预报天数，默认 3

    返回:
        dict: 包含城市信息、坐标、时区与逐日预报列表；
              逐日字段含日期、天气状况、最高/最低温度、体感温度、
              湿度、风速、降水概率与降水量等
    """
    loc = _geocode(city)
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "apparent_temperature_max,apparent_temperature_min,"
                "weather_code,precipitation_probability_max,precipitation_sum,"
                "wind_speed_10m_max,relative_humidity_2m_mean"
            ),
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}

    days_list = []
    for i, date in enumerate(daily.get("time") or []):
        days_list.append(
            {
                "date": date,
                "weather": _describe_weather(daily.get("weather_code", [None])[i]),
                "weather_code": daily.get("weather_code", [None])[i],
                "temp_max_c": daily.get("temperature_2m_max", [None])[i],
                "temp_min_c": daily.get("temperature_2m_min", [None])[i],
                "feels_like_max_c": daily.get("apparent_temperature_max", [None])[i],
                "feels_like_min_c": daily.get("apparent_temperature_min", [None])[i],
                "humidity_pct": daily.get("relative_humidity_2m_mean", [None])[i],
                "wind_speed_max_mps": daily.get("wind_speed_10m_max", [None])[i],
                "precip_prob_pct": daily.get("precipitation_probability_max", [None])[i],
                "precipitation_mm": daily.get("precipitation_sum", [None])[i],
            }
        )

    return {
        "city": loc["name"],
        "country": loc.get("country"),
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": data.get("timezone"),
        "forecast_days": days,
        "daily": days_list,
    }


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "Beijing"
    if len(sys.argv) >= 3 and sys.argv[2] == "forecast":
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        print(json.dumps(get_daily_forecast(city, days), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(get_weather(city), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
