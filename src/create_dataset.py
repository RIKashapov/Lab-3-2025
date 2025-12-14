from clearml import Dataset
from pathlib import Path

DATA_DIR = Path("data/raw")

ds = Dataset.create(
    dataset_project="Lab3",
    dataset_name="weather_history_daily_moscow"
)

ds.add_files(str(DATA_DIR))
ds.upload()
ds.finalize()

print("Dataset ID:", ds.id)