import os
import numpy as np
USER_DIR = os.path.expanduser("~")
DOCUMENTS_DIR = os.path.join(USER_DIR,"Documents")
ATRAIN_DIR = os.path.join(DOCUMENTS_DIR,"aTrain")
MODELS_DIR = os.path.join(ATRAIN_DIR, "models")
TRANSCRIPT_DIR = os.path.join(ATRAIN_DIR,"transcriptions")
METADATA_FILENAME = "metadata.txt"
LOG_FILENAME = "log.txt"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
SAMPLING_RATE = 16000


# audio lengths and embedding/segmentation steps for the step estimation through quadratic regression
audio_lengths = np.array([32, 109, 139, 811, 1320])
segmentation_steps = np.array([56, 211, 270, 1614, 2633])
embedding_steps = np.array([6, 20, 26, 152, 247])
