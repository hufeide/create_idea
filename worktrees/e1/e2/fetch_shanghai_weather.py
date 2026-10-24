"""一次性获取上海市当前天气 + 未来 3 天预报，输出合并的结构化 JSON。"""

import json

from weather_module import get_weather, get_daily_forecast

CITY = "上海"


def main() -> None:
    current = get_weather(CITY)
    forecast = get_daily_forecast(CITY, days=3)

    merged = {
        "source": "Open-Meteo API",
        "city": current["city"],
        "country": current["country"],
        "coordinates": {
            "latitude": current["latitude"],
            "longitude": current["longitude"],
        },
        "current_weather": {
            "observed_at": current["observed_at"],
            "temperature_c": current["temperature_c"],
            "feels_like_c": current["feels_like_c"],
            "humidity_pct": current["humidity_pct"],
            "wind_speed_mps": current["wind_speed_mps"],
            "precipitation_mm": current["precipitation_mm"],
            "weather_code": current["weather_code"],
        },
        "daily_forecast": forecast["daily"],
        "timezone": forecast["timezone"],
    }

    print(json.dumps(merged, ensure_ascii=False, indent=2))
    with open("shanghai_weather.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
