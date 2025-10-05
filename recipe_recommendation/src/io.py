import os
import json
from huggingface_hub import hf_hub_download

# Hugging Face ID
REPO_ID = "Iris314/recipe-cleaned"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(filename: str) -> str:

    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Hugging Face Hub...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False
        )
    else:
        print(f"{filename} already exists locally.")
    return local_path


def load_recipes_csv() -> str:
    return download_file("recipes.csv")


def load_ingredient_map() -> dict:
    path = download_file("ingredient_map.data")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
