#!/usr/bin/env python3
"""
天气 API 模块 - 调用天气 API，按城市返回结构化天气数据
"""

import requests
from datetime import datetime
from typing import Dict, List, Optional


class WeatherAPI:
    """天气 API 类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化天气 API
        
        Args:
            api_key: API 密钥（可选，使用 open-meteo 免费 API 无需密钥）
        """
        self.api_key = api_key
        # 使用 open-meteo 免费天气 API
        self.base_url = "https://api.open-meteo.com/v1/forecast"
    
    def get_weather_by_city(self, city: str, country_code: str = "CN", 
                           params: Optional[Dict] = None) -> Dict:
        """
        按城市获取天气数据
        
        Args:
            city: 城市名称
            country_code: 国家代码（例如：CN 中国，US 美国）
            params: 额外请求参数
            
        Returns:
            包含天气数据的字典
        """
        # 构建城市坐标（使用 open-meteo 的 geocoding API）
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&country={country_code}&count=1&language=zh&format=json"
        
        try:
            response = requests.get(geo_url, timeout=10)
            response.raise_for_status()
            geo_data = response.json()
            
            if not geo_data.get("results"):
                return {
                    "error": f"未找到城市：{city}",
                    "city": city,
                    "country": country_code
                }
            
            # 获取城市坐标
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            place_name = geo_data["results"][0]["name"]
            admin1 = geo_data["results"][0"].get("admin1", "")
            
            # 获取天气数据
            weather_url = f"{self.base_url}?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m,wind_direction_10m&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m,wind_direction_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,weather_code,wind_speed_10m_max&timezone=Asia%2FShanghai"
            
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            # 结构化数据
            return self._structure_weather_data(weather_data, city, country_code)
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API 请求失败：{str(e)}",
                "city": city,
                "country": country_code
            }
    
    def _structure_weather_data(self, data: Dict, city: str, country: str) -> Dict:
        """
        结构化天气数据
        
        Args:
            data: API 返回的原始数据
            city: 城市名称
            country: 国家代码
            
        Returns:
            结构化的天气数据字典
        """
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})
        
        # 当前天气
        current_weather = {
            "temperature_2m": current.get("temperature_2m", "N/A"),
            "relative_humidity": current.get("relative_humidity", "N/A"),
            "apparent_temperature": current.get("apparent_temperature", "N/A"),
            "is_day": current.get("is_day", 0),
            "precipitation": current.get("precipitation", 0),
            "rain": current.get("rain", 0),
            "showers": current.get("showers", 0),
            "snowfall": current.get("snowfall", 0),
            "weather_code": current.get("weather_code", 0),
            "wind_speed_10m": current.get("wind_speed_10m", 0),
            "wind_direction_10m": current.get("wind_direction_10m", 0)
        }
        
        # 结构化返回数据
        return {
            "city": city,
            "country": country,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_weather": current_weather,
            "hourly": {
                "temperature_2m": hourly.get("temperature_2m", []),
                "wind_speed_10m": hourly.get("wind_speed_10m", [])
            },
            "daily": {
                "temperature_2m_max": daily.get("temperature_2m_max", []),
                "temperature_2m_min": daily.get("temperature_2m_min", []),
                "precipitation_sum": daily.get("precipitation_sum", []),
                "wind_speed_10m_max": daily.get("wind_speed_10m_max", [])
            }
        }
    
    def get_weather_codes_meaning(self, weather_code: int) -> str:
        """
        获取天气代码含义
        
        Args:
            weather_code: WMO 天气代码
            
        Returns:
            天气代码的中文描述
        """
        codes = {
            0: "晴朗",
            1: "主要晴朗",
            2: "部分多云",
            3: "多云",
            45: "雾",
            48: "雾凇",
            51: "轻雾",
            53: "中度薄雾",
            55: "浓雾",
            61: "轻微毛毛雨",
            63: "中度毛毛雨",
            65: "大到暴雨",
            66: "冻毛毛雨",
            67: "冻雨",
            71: "轻微雪",
            73: "中雪",
            75: "大雪",
            77: "雪粒",
            80: "轻微降雨",
            81: "中雨",
            82: "暴雨",
            85: "轻微小雪",
            86: "中雪",
            95: "雷电",
            96: "雷电伴轻微小雪",
            99: "雷电伴中雪"
        }
        return codes.get(weather_code, f"未知代码：{weather_code}")


def main():
    """测试函数"""
    api = WeatherAPI()
    
    # 测试城市
    cities = ["北京", "上海", "广州", "深圳"]
    
    print("=" * 60)
    print("天气 API 测试 - 按城市返回结构化天气数据")
    print("=" * 60)
    
    for city in cities:
        print(f"\n📍 城市：{city}")
        print("-" * 40)
        
        weather = api.get_weather_by_city(city, "CN")
        
        if "error" in weather:
            print(f"❌ 错误：{weather['error']}")
        else:
            current = weather["current_weather"]
            print(f"🌡️  当前温度：{current['temperature_2m']}°C")
            print(f"💧 湿度：{current['relative_humidity']}%")
            print(f"💨 风速：{current['wind_speed_10m']} m/s")
            print(f"☁️  天气：{api.get_weather_codes_meaning(current['weather_code'])}")
            print(f"⏰ 时间：{weather['current_time']}")


if __name__ == "__main__":
    main()
