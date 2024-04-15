from importlib.resources import files
from huggingface_hub import snapshot_download
import shutil
import requests
import json
import os
from tqdm import tqdm
import platform
from .globals import ATRAIN_DIR

def download_all_resources():
    """Downloads all resources including models."""
    download_all_models()

def download_all_models():
    """Downloads all models defined in the model configuration file."""
    models_config = load_model_config_file()
    for model in models_config:
        get_model(model)

def load_model_config_file():
    """Loads the model configuration file.

    Returns:
        dict: Dictionary containing model configurations.
    """

    # only load large v3
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    with open(models_config_path, "r") as models_config_file:
        models_config = json.load(models_config_file)
    return models_config

def get_model(model):
    """Loads a specific model.

    Args:
        model (str): Name of the model to load.

    Returns:
        str: Path to the downloaded model.
    """
    
    models_config = load_model_config_file()
    model_info = models_config[model]
    model_path = os.path.join(ATRAIN_DIR, "models", model)
    if not os.path.exists(model_path):
        snapshot_download(repo_id=model_info["repo_id"], revision=model_info["revision"], local_dir=model_path, local_dir_use_symlinks=False)
        print(f"Model downloaded to {model_path}")
    return model_path


if __name__ == "__main__":
    ...