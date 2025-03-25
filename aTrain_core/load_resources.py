import json
import os
import shutil
from functools import partial
from importlib.resources import files

import huggingface_hub
from tqdm.auto import tqdm

from .globals import MODELS_DIR, REQUIRED_MODELS
from .GUI_integration import EventSender


class custom_tqdm(tqdm):
    def __init__(self, GUI: EventSender, total: float, *args, **kwargs):
        self.GUI = GUI
        super().__init__(total=total, *args, **kwargs)

    def update(self, n=1):
        current = self.n + n
        self.GUI.progress_info(current, self.total)
        super().update(n)


def download_all_models():
    """Downloads all models defined in the model configuration file."""
    models_config = load_model_config_file()
    for model in models_config:
        get_model(model)


def get_model(
    model: str,
    GUI: EventSender = EventSender(),
    models_dir=MODELS_DIR,
    required_models_dir=MODELS_DIR,
) -> str:
    """Loads a specific model."""
    models_config = load_model_config_file()
    model_info = models_config[model]
    models_dir = required_models_dir if model in REQUIRED_MODELS else models_dir
    model_path = os.path.join(models_dir, model)

    if os.path.exists(model_path):
        return model_path

    repo_size = model_info["repo_size"]
    tqdm_bar = custom_tqdm(total=repo_size, GUI=GUI)

    # Monkey patching custom tqdm bar into the huggingface snapshot download
    huggingface_hub.file_download.http_get = partial(
        huggingface_hub.file_download.http_get, _tqdm_bar=tqdm_bar
    )

    huggingface_hub.snapshot_download(
        repo_id=model_info["repo_id"],
        revision=model_info["revision"],
        local_dir=model_path,
        local_dir_use_symlinks=False,
        max_workers=1,
    )

    return model_path


def remove_model(model, models_dir=MODELS_DIR):
    model_path = os.path.join(models_dir, model)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # This deletes the directory and all its contents


def load_model_config_file() -> dict:
    """Loads the model configuration file."""
    models_config_path = str(files("aTrain_core.data").joinpath("models.json"))
    with open(models_config_path, "r") as models_config_file:
        models_config: dict = json.load(models_config_file)
    return models_config


if __name__ == "__main__":
    ...
