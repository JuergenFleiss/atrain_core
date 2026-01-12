from importlib.resources import files
from importlib.util import find_spec
from platformdirs import user_documents_path

ATRAIN_DIR = user_documents_path() / "aTrain"
MODELS_DIR = ATRAIN_DIR / "models"
REQUIRED_MODELS_DIR = (
    files("aTrain") / "required_models" if find_spec("aTrain") else MODELS_DIR
)
REQUIRED_MODELS = ["speaker-detection", "large-v3-turbo"]
TRANSCRIPT_DIR = ATRAIN_DIR / "transcriptions"
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000
