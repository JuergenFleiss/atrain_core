import json
import os

# from huggingface_hub import snapshot_download
import shutil
from importlib.resources import files

from .custom_snapshot_download import snapshot_download
from .globals import MODELS_DIR
from .GUI_integration import EventSender, ProgressTracker
from .step_estimator import get_total_model_download_steps


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


def get_model(model: str, GUI: EventSender = None, models_dir=MODELS_DIR) -> str:
    if GUI is None:
        GUI = EventSender()

    """Loads a specific model."""
    models_config = load_model_config_file()
    model_info = models_config[model]
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        total_chunks = get_total_model_download_steps(model)
        tracker = ProgressTracker(total_chunks)

        # Define a callback function for progress tracking
        def progress_callback(current_chunk):
            progress_info = tracker.progress_callback(current_chunk)
            GUI.progress_info(
                current=progress_info["current"], total=progress_info["total"]
            )

        snapshot_download(
            repo_id=model_info["repo_id"],
            revision=model_info["revision"],
            local_dir=model_path,
            local_dir_use_symlinks=False,
            progress_callback=progress_callback,
        )
        print(f"Model downloaded to {model_path}")

    return model_path


def remove_model(model, models_dir=MODELS_DIR):
    model_path = os.path.join(models_dir, model)
    print(f"Removing model {model} at path: {model_path}")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)  # This deletes the directory and all its contents


if __name__ == "__main__":
    ...
