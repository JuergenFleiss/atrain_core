from importlib.resources import files
from huggingface_hub import snapshot_download as original_snapshot_download, hf_hub_download
import shutil
from typing import Optional, Callable
import json
import os
from tqdm.auto import tqdm as base_tqdm
import platform
from .globals import ATRAIN_DIR

system = platform.system()

class CustomHuggingFaceHub:
    def __init__(self, repo_id, **kwargs):
        self.repo_id = repo_id
        self.kwargs = kwargs

    def _inner_hf_hub_download_with_print(self, repo_file: str, callback: Optional[Callable] = None):
        result = hf_hub_download(
            self.repo_id,
            filename=repo_file,
            **self.kwargs
        )
        print(f"Downloaded: {repo_file}")
        if callback:
            callback(repo_file)
        return result
    
    


class ModelDownloadProgress(base_tqdm):
    def __init__(self, *args, update_callback=None, **kwargs):
        self.update_callback = update_callback
        super().__init__(*args, **kwargs)
    
    def update(self, n=1):
        super().update(n)
        if self.update_callback:
            self.update_callback(self.n, self.total)
    
    def set_description(self, desc=None, refresh=True):
        super().set_description(desc, refresh)
        if self.update_callback:
            self.update_callback(self.n, self.total)

def snapshot_download(
    *args,
    progress_callback=None,
    **kwargs
) -> str:
    custom_tqdm_class = ModelDownloadProgress if progress_callback else None
    return original_snapshot_download(
        *args,
        tqdm_class=custom_tqdm_class,
        **kwargs
    )

def progress_update(current, total):
    progress = {
        "current": current,
        "total": total,
        "percentage": (current / total) * 100 if total else 0,
    }
    print(f"PROGRESS: {progress}")  # Replace this with actual logic

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
        snapshot_download(
            repo_id=model_info["repo_id"],
            revision=model_info["revision"],
            local_dir=model_path,
            local_dir_use_symlinks=False,
            progress_callback=lambda file: print(f"Completed downloading: {file}")
        )
        print(f"Model downloaded to {model_path}")
    return model_path

def remove_model(model):
    if system == "Linux":
        model_path = os.path.join(ATRAIN_DIR, "models", model)
    if system in ["Windows", "Darwin"]:
        model_path = os.path.join(ATRAIN_DIR, "models", model)
    print(f"Removing model {model} at path: {model_path}")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # This deletes the directory and all its contents

if __name__ == "__main__":
    ...
