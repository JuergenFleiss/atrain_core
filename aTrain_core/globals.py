import os

USER_DIR = os.path.expanduser("~")
DOCUMENTS_DIR = os.path.join(USER_DIR, "Documents")
ATRAIN_DIR = os.path.join(DOCUMENTS_DIR, "aTrain")
MODELS_DIR = os.path.join(ATRAIN_DIR, "models")
TRANSCRIPT_DIR = os.path.join(ATRAIN_DIR, "transcriptions")
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000
REQUIRED_MODELS = ["diarize", "large-v3-turbo"]
