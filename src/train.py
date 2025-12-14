import json
from pathlib import Path
import numpy as np
import pandas as pd
from clearml import Task, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

TARGET = "temp_avg"

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    # сезонность
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df

def add_lags(df: pd.DataFrame, col: str, lags=(1, 7, 14), rolls=(7, 14)) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
    for w in rolls:
        df[f"{col}_rollmean{w}"] = (
            df.groupby("city")[col].shift(1).rolling(window=w).mean().reset_index(level=0, drop=True)
        )
    return df

def make_supervised(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # y_{t+h}
    df = df.sort_values(["city", "date"]).copy()
    df[f"y_h{horizon}"] = df.groupby("city")[TARGET].shift(-horizon)
    return df

def main():
    task = Task.init(project_name="Lab3", task_name="train_catboost_7d")
    task.output_uri = "http://localhost:8081"
    logger = task.get_logger()

    params = {
        "dataset_id": "dc7eda11b34a48e2ae4b8f69f8ee55cf",
        "target": TARGET,
        "horizons": [1,2,3,4,5,6,7],
        "train_frac": 0.8,
        "model_depth": 6,
        "model_learning_rate": 0.08,
        "model_iterations": 800,
        "model_l2_leaf_reg": 3,
        "random_seed": 42,
    }
    task.connect(params)

    ds = Dataset.get(dataset_id=params["dataset_id"], alias="weather_history_daily_moscow")
    data_path = Path(ds.get_local_copy()) / "daily.csv"
    df = pd.read_csv(data_path)

    df = add_time_features(df)
    df = add_lags(df, TARGET)

    # выбросим строки где нет лагов
    df = df.dropna().reset_index(drop=True)

    # временное разбиение без shuffle
    df = df.sort_values("date")
    split = int(len(df) * params["train_frac"])
    df_train_base = df.iloc[:split].copy()
    df_val_base = df.iloc[split:].copy()

    features = [c for c in df.columns if c not in ["date", "city", "temp_max", "temp_min", "temp_avg", "precipitation_sum"]]
    # добавим city как категорию
    features = ["city"] + features

    run_dir = Path("artifacts") / task.id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir

    out_dir.mkdir(exist_ok=True)

    metrics_table = []

    for h in params["horizons"]:
        dtrain = make_supervised(df_train_base, h).dropna()
        dval = make_supervised(df_val_base, h).dropna()

        Xtr, ytr = dtrain[features], dtrain[f"y_h{h}"]
        Xva, yva = dval[features], dval[f"y_h{h}"]

        model = CatBoostRegressor(
            depth=int(params["model_depth"]),
            learning_rate=float(params["model_learning_rate"]),
            iterations=int(params["model_iterations"]),
            l2_leaf_reg=float(params["model_l2_leaf_reg"]),
            loss_function="RMSE",
            random_seed=int(params["random_seed"]),
            verbose=False,
        )

        model.fit(Xtr, ytr, cat_features=[0])  # "city" в колонке 0

        pred = model.predict(Xva)

        mae = mean_absolute_error(yva, pred)
        r = rmse(yva, pred)
        mp = smape(yva, pred)

        logger.report_scalar("metrics", f"MAE_h{h}", iteration=0, value=mae)
        logger.report_scalar("metrics", f"RMSE_h{h}", iteration=0, value=r)
        logger.report_scalar("metrics", f"sMAPE_h{h}", iteration=0, value=mp)

        metrics_table.append({"h": h, "MAE": mae, "RMSE": r, "sMAPE": mp})

        # сохраним модель
        model_path = out_dir / f"model_h{h}.cbm"
        model.save_model(str(model_path))

        # feature importance как артефакт (простая таблица)
        fi = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
        fi_path = out_dir / f"fi_h{h}.csv"
        fi.to_csv(fi_path, index=False)

    # итоговая таблица метрик
    mt = pd.DataFrame(metrics_table)
    avg_rmse = float(mt["RMSE"].mean())
    logger.report_scalar("metrics", "RMSE_avg_7d", iteration=0, value=avg_rmse)

    mt_path = out_dir / "metrics_by_horizon.csv"
    mt.to_csv(mt_path, index=False)
    task.upload_artifact("metrics_by_horizon", str(mt_path))

    avg_rmse = float(mt["RMSE"].mean())
    logger.report_scalar("metrics", "RMSE_avg_7d", iteration=0, value=avg_rmse)
    print("RMSE_avg_7d:", avg_rmse)

    # график RMSE по горизонту
    plt.figure()
    plt.plot(mt["h"], mt["RMSE"], marker="o")
    plt.xlabel("Horizon (days)")
    plt.ylabel("RMSE")
    plt.title("RMSE by horizon")
    plot_path = out_dir / "rmse_by_horizon.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    task.upload_artifact("rmse_plot", str(plot_path))

    # сохраним конфиг
    cfg_path = out_dir / "train_config.json"
    cfg_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    task.upload_artifact("config", str(cfg_path))

    print(mt)

if __name__ == "__main__":
    main()
