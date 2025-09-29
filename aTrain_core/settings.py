import json
import os
from importlib.resources import files
from .load_resources import load_model_config_file
from enum import StrEnum, auto


class Device(StrEnum):
    CPU = auto()
    CUDA = auto()


class ComputeType(StrEnum):
    INT8 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()


def check_inputs_transcribe(file, model, language, device):
    """Check the validity of inputs for the transcription process."""

    file_correct = check_file(file)
    model_correct = check_model(model, language)
    language_correct = check_language(language)

    if not file_correct and model_correct and language_correct:
        raise ValueError(
            "Incorrect input. Please check the file, model and language inputs."
        )


def load_formats() -> list:
    formats_json_path = str(files("aTrain_core.data").joinpath("formats.json"))
    with open(formats_json_path, "r") as f:
        file_formats: list = json.load(f)
    return file_formats


def check_file(filename):
    """Check if the provided file is in a correct format for transcription."""
    file_extension = os.path.splitext(filename)[-1]
    file_extension_lower = str(file_extension).lower()
    correct_file_formats = load_formats()
    return file_extension_lower in correct_file_formats


def check_device(device):
    if device == "GPU":
        from torch import cuda

        cuda_available = cuda.is_available()
        if cuda_available:
            return device
        else:
            raise ValueError(
                "GPU is not available. Please choose --device CPU instead."
            )


def check_model(model, language):
    """Check if the provided model and language are valid for transcription."""

    all_model_configs = load_model_config_file()
    all_models = set(all_model_configs.keys())
    all_models.remove("diarize")

    model_available = model in all_models

    if not model_available:
        raise ValueError(
            f"Model {model} is not available. These are the available models: {all_models}"
        )

    model_config: dict = all_model_configs[model]

    if "languages" in model_config.keys():
        model_languages: list = model_config["languages"]
    else:
        model_languages = load_languages().keys()

    if language not in model_languages:
        raise ValueError(
            f"Language input wrong or unspecified. This model is only available in {model_languages} and has to be specified."
        )

    return model_available


def load_languages() -> dict:
    languages_json_path = str(files("aTrain_core.data").joinpath("languages.json"))
    with open(languages_json_path, "r") as f:
        languages = json.load(f)
    return languages


def check_language(language):
    """Check if the provided language is supported for transcription."""
    languages_json = load_languages()
    correct_languages = languages_json.keys()
    return language in correct_languages
