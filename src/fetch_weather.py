import requests
import pandas as pd
from datetime import datetime

cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867)
}

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

all_data = []

for city, (lat, lon) in cities.items():

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=Asia%2FKolkata"
    )

    response = requests.get(url)
    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "Date": daily["time"],
        "City": city,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"]
    })

    all_data.append(df)

weather_df = pd.concat(all_data)
weather_df.to_csv("data/raw/weather_data.csv", index=False)

print("Weather data saved.")