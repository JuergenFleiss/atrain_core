import os
from importlib.resources import files


USER_DIR = os.path.expanduser("~")
DOCUMENTS_DIR = os.path.join(USER_DIR, "Documents")
ATRAIN_DIR = os.path.join(DOCUMENTS_DIR, "aTrain")
MODELS_DIR = os.path.join(ATRAIN_DIR, "models")
REQUIRED_MODELS_DIR = files("aTrain") / "required_models"
REQUIRED_MODELS = ["diarize", "large-v3-turbo"]
TRANSCRIPT_DIR = os.path.join(ATRAIN_DIR, "transcriptions")
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000
