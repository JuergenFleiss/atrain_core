from importlib.resources import files
from huggingface_hub import snapshot_download
import shutil
import requests
import json
import os
from tqdm import tqdm
import platform

def download_all_resources():
    download_all_models()

def download_all_models():
    models_config = load_model_config_file()
    for model in models_config:
        print(model)
        get_model(model)

def load_model_config_file():
    # only load large v3
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    print(f"load_model_config_file: Model config path: {models_config_path}")
    with open(models_config_path, "r") as models_config_file:
        models_config = json.load(models_config_file)
    return models_config

def get_model(model): # loads only one model
    models_config = load_model_config_file()
    model_info = models_config[model]
    model_path = str(files("aTrain_core.models").joinpath(model))
    print(f"get_model model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"REPO ID: {model_info['repo_id']}")
        snapshot_download(repo_id=model_info["repo_id"], revision=model_info["revision"], local_dir=model_path, local_dir_use_symlinks=False)
        print("snapshot download of repo by id")
    return model_path


if __name__ == "__main__":
    ...