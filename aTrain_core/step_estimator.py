import json
import math
from importlib.resources import files


def load_model_config_file():
    """Loads the model configuration file.
    Function defined again to avoid circular import error"""

    # only load large v3
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    with open(models_config_path, "r") as models_config_file:
        models_config = json.load(models_config_file)
    return models_config


def calculate_steps(audio_duration):
    """Calculates the total number of steps for the transcription process."""
    # Predictions
    segmentation_prediction = predict_segmentation_steps(audio_duration)
    embedding_prediction = predict_embedding_steps(audio_duration)
    segmentation_prediction = segmentation_prediction / 32
    # account for one extra process when not divisible by 32
    if isinstance(segmentation_prediction / 32, int):
        segmentation_prediction = segmentation_prediction
    else:
        segmentation_prediction = math.ceil(segmentation_prediction + 1)

    total_steps = (
        segmentation_prediction + embedding_prediction + 2
    )  # speaker_counting & discrete_diarization are one step each
    return total_steps


def predict_segmentation_steps(length):
    """A function that estimates the number of computational steps during the segmentation process."""
    # These coefficients come from running a quadratic regression on data form some sample files.
    a = 1959430015971981 / 1180591620717411303424
    b = 4499524351105379 / 2251799813685248
    c = -1060416894995783 / 140737488355328
    return a * length**2 + b * length + c


def predict_embedding_steps(length):
    """A function that estimates the number of computational steps during the embedding process."""
    # These coefficients come from running a quadratic regression on data form some sample files.
    a = -6179382659256857 / 9444732965739290427392
    b = 6779072611703841 / 36028797018963968
    c = -3802017452395983 / 18014398509481984
    return a * length**2 + b * length + c


def get_total_model_download_steps(model_name):
    """A function that finds the total download chunks (steps) for a given model.
    The metadata has been pre-calculated by downloading the models"""

    models_config = load_model_config_file()
    model_info = models_config[model_name]

    if model_name in models_config:
        total_chunks = model_info["chunks_total"]

    else:
        total_chunks = None

    return total_chunks
