from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from clearml import Model
import os
import requests
from clearml.backend_api.session import Session

TARGET = "temp_avg"

FEATURES = [
    "city",
    "dow", "doy", "sin_doy", "cos_doy",
    "temp_avg_lag1",
    "temp_avg_lag7",
    "temp_avg_lag14",
    "temp_avg_rollmean7",
    "temp_avg_rollmean14",
]

HORIZONS = [1, 2, 3, 4, 5, 6, 7]

app = FastAPI(title="Weather Forecast API")

MODELS: Dict[int, CatBoostRegressor] = {}
HISTORY: Optional[pd.DataFrame] = None


class PredictRequest(BaseModel):
    city: str = Field(..., examples=["Moscow"])
    dates: List[str] = Field(
        ...,
        description="List of dates D+1..D+7 (7 items)",
        examples=[["2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22"]],
    )


def _rewrite_fileserver_url(url: str) -> str:
    files_host = os.getenv("CLEARML_FILES_HOST")
    if files_host and url.startswith("http://localhost:8081/"):
        return url.replace("http://localhost:8081", files_host.rstrip("/"))

    return url.replace("http://localhost:8081", "http://host.docker.internal:8081")


def download_model_file(model_url: str, dst_path: Path) -> Path:
    url = _rewrite_fileserver_url(model_url)

    token = Session().token
    headers = {
        "Authorization": f"Bearer {token}",
        "X-ClearML-Auth-Token": token,
    }

    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(r.content)
    return dst_path


def add_time_features_for_date(d: pd.Timestamp) -> dict:
    doy = int(d.dayofyear)
    return {
        "dow": int(d.dayofweek),
        "doy": doy,
        "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
    }


def build_feature_row(history_city: pd.DataFrame, city: str, d: pd.Timestamp) -> dict:
    history_city = history_city.sort_values("date")

    def get_value(date_: pd.Timestamp):
        s = history_city.loc[history_city["date"] == date_, TARGET]
        return None if s.empty else float(s.iloc[0])

    def get_roll_mean(window: int):
        start = d - pd.Timedelta(days=window)
        end = d - pd.Timedelta(days=1)
        mask = (history_city["date"] >= start) & (history_city["date"] <= end)
        vals = history_city.loc[mask, TARGET].astype(float)
        return None if len(vals) < window else float(vals.mean())

    lag1 = get_value(d - pd.Timedelta(days=1))
    lag7 = get_value(d - pd.Timedelta(days=7))
    lag14 = get_value(d - pd.Timedelta(days=14))
    roll7 = get_roll_mean(7)
    roll14 = get_roll_mean(14)

    missing = []
    for name, v in [("lag1", lag1), ("lag7", lag7), ("lag14", lag14), ("roll7", roll7), ("roll14", roll14)]:
        if v is None:
            missing.append(name)

    if missing:
        raise ValueError(f"Not enough history for city={city} date={d.date()} missing={missing}")

    row = {"city": city}
    row.update(add_time_features_for_date(d))
    row.update({
        "temp_avg_lag1": lag1,
        "temp_avg_lag7": lag7,
        "temp_avg_lag14": lag14,
        "temp_avg_rollmean7": roll7,
        "temp_avg_rollmean14": roll14,
    })
    return row


def _extract_horizon_from_name(name: str) -> Optional[int]:
    if "model_h" not in name:
        return None
    try:
        return int(name.split("model_h")[-1].strip())
    except Exception:
        return None


@app.on_event("startup")
def startup():
    global MODELS, HISTORY

    # 1) history
    data_path = Path("data/raw/daily.csv")
    if not data_path.exists():
        raise RuntimeError(f"History file not found: {data_path}. Make sure it's inside the image at /app/data/raw/daily.csv")

    HISTORY = pd.read_csv(data_path)
    HISTORY["date"] = pd.to_datetime(HISTORY["date"])

    # 2) models
    MODELS.clear()

    prod_models = Model.query_models(only_published=True, project_name="Lab3")

    # соберём только model_h1..model_h7
    selected: Dict[int, Model] = {}
    for m in prod_models:
        h = _extract_horizon_from_name(getattr(m, "name", "") or "")
        if h in HORIZONS:
            # если вдруг несколько моделей на один h — оставим последнюю найденную
            selected[h] = m

    if set(selected.keys()) != set(HORIZONS):
        raise RuntimeError(f"Expected production models for horizons {HORIZONS}, found {sorted(selected.keys())}")

    models_dir = Path("/app/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        m = selected[h]
        if not m.url:
            raise RuntimeError(f"Model h{h} id={m.id} has empty url")

        local_path = download_model_file(m.url, models_dir / f"model_h{h}.cbm")

        cb = CatBoostRegressor()
        cb.load_model(str(local_path))
        MODELS[h] = cb

    print("Startup complete: models loaded =", sorted(MODELS.keys()))


@app.post("/predict")
def predict(req: PredictRequest):
    if HISTORY is None:
        raise HTTPException(500, "History not loaded")

    if len(req.dates) != 7:
        raise HTTPException(400, f"Expected 7 dates (D+1..D+7), got {len(req.dates)}")

    city = req.city
    dates = [pd.to_datetime(d) for d in req.dates]

    # базовая история по городу
    hist = HISTORY.loc[HISTORY["city"] == city, ["date", TARGET]].copy()
    if hist.empty:
        raise HTTPException(400, f"Unknown city: {city}")

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    predictions = []

    # Рекурсивно: предсказали день -> добавили в историю -> следующий день
    for i, d in enumerate(dates, start=1):
        model = MODELS.get(i)
        if model is None:
            raise HTTPException(500, f"Model for horizon h{i} not loaded")

        try:
            row = build_feature_row(hist, city, d)
        except ValueError as e:
            raise HTTPException(400, str(e))

        X = pd.DataFrame([row])[FEATURES]
        y = float(model.predict(X)[0])

        predictions.append({"date": str(d.date()), "horizon": i, "temp_avg": y})

        # важно: добавляем прогноз в историю, чтобы лаги/роллы были доступны дальше
        hist = pd.concat([hist, pd.DataFrame([{"date": d, TARGET: y}])], ignore_index=True)

    return {"city": city, "predictions": predictions}