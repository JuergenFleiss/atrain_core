from importlib.resources import files
#from huggingface_hub import snapshot_download
import shutil
import json
import os
from .custom_snapshot_download import snapshot_download
from .globals import MODELS_DIR
from .GUI_integration import EventSender

def download_all_models():
    """Downloads all models defined in the model configuration file."""
    models_config = load_model_config_file()
    for model in models_config:
        get_model(model)

def load_model_config_file():
    """Loads the model configuration file."""

    # only load large v3
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    with open(models_config_path, "r") as models_config_file:
        models_config = json.load(models_config_file)
    return models_config

def get_model(model: str , GUI = EventSender()) -> str:
    """Loads a specific model."""
    models_config = load_model_config_file()
    model_info = models_config[model]
    model_path = os.path.join(MODELS_DIR, model)
    if not os.path.exists(model_path):
        snapshot_download(repo_id=model_info["repo_id"], revision=model_info["revision"], local_dir=model_path, local_dir_use_symlinks=False)
        print(f"Model downloaded to {model_path}")
    return model_path


def remove_model(model):
    model_path = os.path.join(MODELS_DIR, model)
    print(f"Removing model {model} at path: {model_path}")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # This deletes the directory and all its contents
        


if __name__ == "__main__":
    ...