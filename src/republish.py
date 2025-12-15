import os
from clearml import Task, Model, OutputModel

PROD_FILE_MODELS = {
    1: "8585de69a04b41fd9cc2b9cf9650bc07",
    2: "ec30856597154b41925d9412cdb98065",
    3: "372edcc375bd4f5f8d1dcc7cc97229b5",
    4: "336ce0d88c304ec987d7345c51822fbf",
    5: "151d97ee2cec4a7f9d6d1e4423a74a09",
    6: "78c3f6fba70946e6b656d887ae73246c",
    7: "23ecf3ac8d5f4a04b4c22af87d88e4b9",
}

OUTPUT_URI = "http://localhost:8081"

def resolve_local_path(m: Model) -> str:
    # 1) пробуем стандартно
    p = m.get_local_copy()
    if p and os.path.exists(p):
        return p

    # 2) fallback: если url=file://..., пробуем как путь
    if (m.url or "").startswith("file://"):
        p2 = m.url.replace("file://", "")
        if os.path.exists(p2):
            return p2

    raise FileNotFoundError(f"Cannot resolve local file for model_id={m.id}, url={m.url}")

def main():
    task = Task.init(project_name="Lab3", task_name="republish_prod_models_to_localhost_fileserver")
    task.output_uri = OUTPUT_URI

    new_ids = {}
    for h, mid in PROD_FILE_MODELS.items():
        src = Model(model_id=mid)
        local_path = resolve_local_path(src)

        out = OutputModel(
            task=task,
            name=f"prod_weather_catboost_7d_h{h}",
            framework="CatBoost",
            tags=["production_http", "best", "series_weather_7d", "temp_avg", "moscow", f"h{h}"],
        )
        out.update_weights(weights_filename=local_path)
        out.publish()
        new_ids[h] = out.id
        print(f"h{h}: NEW_ID={out.id} URL={out.url}")

    task.close()

    print("\nNEW production_http model ids:")
    for h in range(1, 8):
        print(h, new_ids[h])

if __name__ == "__main__":
    main()
