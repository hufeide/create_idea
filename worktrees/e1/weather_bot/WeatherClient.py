"""Weather client for handling HTTP requests to Open-Meteo API."""

import urllib.request
import urllib.error
import urllib.parse
import json


def get_coordinates(city: str) -> tuple[float, float] | None:
    """
    Get coordinates for a city using Open-Meteo search endpoint.
    
    Args:
        city: City name to search for
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    # Search for city coordinates
    search_url = "https://api.open-meteo.com/v1/search"
    params = urllib.parse.urlencode({"search": city, "count": 1})
    full_url = f"{search_url}?{params}"
    
    try:
        with urllib.request.urlopen(full_url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        if "results" in data and data["results"]:
            result = data["results"][0]
            return (result["latitude"], result["longitude"])
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def fetch_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch weather forecast data for given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Weather data dictionary or None if fetch fails
    """
    # Fetch current weather and daily forecast
    url = "https://api.open-meteo.com/v1/forecast"
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current_weather": "",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min",
        "timezone": "auto"
    })
    full_url = f"{url}?{params}"
    
    try:
        with urllib.request.urlopen(full_url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
