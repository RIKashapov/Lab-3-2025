from pathlib import Path
from clearml import Task, OutputModel

UPLOAD_URI = "http://host.docker.internal:8081"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

def main():
    task = Task.init(project_name="Lab3", task_name="republish_models_to_fileserver")

    task.output_uri = UPLOAD_URI

    for h in range(1, 8):
        src_path = ARTIFACTS_DIR / f"model_h{h}.cbm"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing: {src_path}")

        name = f"weather_catboost_7d_h{h}"

        om = OutputModel(
            task=task,
            name=name,
            label_enumeration=None,
            framework="CatBoost",
            tags=["best", "7d", "catboost", "moscow", "temp_avg", f"h{h}"],
        )

        # загрузит файл в fileserver (через task.output_uri)
        om.update_weights(weights_filename=str(src_path))

        # публикуем в Model Registry
        om.publish()

        print(f"Published {name} -> model_id={om.id}")

    task.close()

if __name__ == "__main__":
    main()