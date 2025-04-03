import os
import sys
from importlib.resources import files


USER_DIR = os.environ['SNAP_USER_DATA']
DOCUMENTS_DIR = USER_DIR
ATRAIN_DIR = os.path.join(DOCUMENTS_DIR, "aTrain")
MODELS_DIR = os.path.join(ATRAIN_DIR, "models")
REQUIRED_MODELS_DIR = os.environ['SNAP_DATA']
REQUIRED_MODELS = ["diarize", "large-v3-turbo"]
TRANSCRIPT_DIR = os.path.join(ATRAIN_DIR, "transcriptions")
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000

