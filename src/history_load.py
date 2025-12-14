import pandas as pd
import requests
from pathlib import Path

CITIES = {
    "Moscow": (55.7558, 37.6173),
    # можно добавить потом:
    # "Samara": (53.2415, 50.2212),
}

START_DATE = "2021-01-01"   # >= 3 года истории
END_DATE   = "2025-12-13"   # вчера относительно 2025-12-14

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def fetch_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    daily = r.json()["daily"]

    return pd.DataFrame({
        "date": daily["time"],
        "city": city,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "temp_avg": daily["temperature_2m_mean"],
        "precipitation_sum": daily["precipitation_sum"],
    })

def main():
    frames = [fetch_city(city, lat, lon) for city, (lat, lon) in CITIES.items()]
    df = pd.concat(frames, ignore_index=True)

    out_path = OUT / "daily.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows:", len(df))
    print(df.head())

if __name__ == "__main__":
    main()
