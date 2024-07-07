import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, TypeVar
from importlib.resources import files
import shutil
import json
from tqdm import tqdm
import platform
from tqdm.auto import tqdm as base_tqdm
from .globals import ATRAIN_DIR
from tqdm.contrib.concurrent import thread_map as original_thread_map
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download as original_snapshot_download

system = platform.system()

U = TypeVar("U")
V = TypeVar("V")

SHARED_THREAD_POOL = ThreadPoolExecutor()

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

def thread_map(
    fn: Callable[[U], V],
    it: Iterable[U],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    unit: str = "it",
    verbose: bool = False,
) -> "list[V]":
    disable = not verbose
    total = total or len(it)
    results = []
    with tqdm(desc=desc, total=total, unit=unit, disable=disable) as pbar:
        futures = SHARED_THREAD_POOL.map(fn, it)
        for result in futures:
            results.append(result)
            pbar.update()
    return results

def custom_snapshot_download(model_info, progress_callback=None, **kwargs):
    """Custom version of snapshot_download with progress callback."""
    if progress_callback is None:
        return original_snapshot_download(repo_id=model_info["repo_id"], revision=model_info["revision"], **kwargs)
    else:
        # Extend snapshot_download with progress callback
        def progress_callback_wrapper(progress):
            progress_callback(progress)
            return original_snapshot_download(repo_id=model_info["repo_id"], revision=model_info["revision"], **kwargs)

        return progress_callback_wrapper

def progress_update(current, total):
    progress = {
        "current": current,
        "total": total,
        "percentage": (current / total) * 100 if total else 0,
    }
    print(f"PROGRESS: {progress}")  # Replace this with actual logic

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
        custom_snapshot_download(
            model_info,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            progress_callback=progress_update,
        )()
        
        print(f"Model downloaded to {model_path}")
    
    return model_path

if __name__ == "__main__":
    model_name = "your_model_name"  # Replace with your desired model name
    model_path = get_model(model_name)
    print(f"Model path: {model_path}")
