from huggingface_hub import hf_hub_download
import shutil
import os

os.makedirs("data", exist_ok=True)

files = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
]

for filename in files:
    path = hf_hub_download(
        repo_id="kashif/M5",
        repo_type="dataset",
        filename=filename,
    )
    shutil.copy(path, os.path.join("data", filename))
    print(f"Saved {filename} to data/")
