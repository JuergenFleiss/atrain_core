from importlib.resources import files
from huggingface_hub import snapshot_download
import shutil
import requests
import json
import os
from tqdm import tqdm
import platform
from check_inputs import check_language

language = "en"

def check_model(model, language):
    # better to look into models.json and check if available
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)
    print(models)
    available_models = []
    for key in models.keys():
        available_models.append(key)
    
    if model not in available_models:
        raise ValueError(f"Model {model} is not available. These are the available models: {available_models} (Note: model 'diarize' is for speaker detection only)")
    
    elif models[model]["language"] != language:
        return model in available_models
    


if __name__ == "__main__":
    check_model("faster-distil-english")
    # download_all_resources()
    # download_all_models()
    # load_model_config_file()
    # get_model(model)