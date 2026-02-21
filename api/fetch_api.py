# api/fetch_api.py

import requests


CITY_MAPPING = {
    "Delhi": "Delhi",
    "Mumbai": "Mumbai",
    "Bangalore": "Bengaluru",
    "Chennai": "Chennai",
    "Hyderabad": "Hyderabad"
}


def fetch_pm25(city_name="Delhi"):

    mapped_city = CITY_MAPPING.get(city_name, city_name)

    url = "https://api.openaq.org/v2/latest"

    params = {
        "city": mapped_city,
        "parameter": "pm25",
        "limit": 1
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        value = data["results"][0]["measurements"][0]["value"]
        return float(value)

    except Exception:
        return None