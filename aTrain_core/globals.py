import os
import sys
from importlib.resources import files


USER_DIR = os.environ['SNAP_USER_DATA']
DOCUMENTS_DIR = os.path.join(USER_DIR, "Documents")
ATRAIN_DIR = os.path.join(DOCUMENTS_DIR, "aTrain")
MODELS_DIR = os.path.join(ATRAIN_DIR, "models")
REQUIRED_MODELS_DIR = MODELS_DIR
REQUIRED_MODELS = []
TRANSCRIPT_DIR = os.path.join(ATRAIN_DIR, "transcriptions")
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000

