from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
from typing import cast

from platformdirs import user_documents_path

ATRAIN_DIR = user_documents_path() / "aTrain"
MODELS_DIR = ATRAIN_DIR / "models"
if find_spec("aTrain"):
    REQUIRED_MODELS_DIR = cast(Path, files("aTrain") / "required_models")
else:
    REQUIRED_MODELS_DIR = MODELS_DIR
REQUIRED_MODELS = ["speaker-detection", "large-v3-turbo"]
TRANSCRIPT_DIR = ATRAIN_DIR / "transcriptions"
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000
