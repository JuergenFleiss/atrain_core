import gc
import os
from datetime import datetime
from multiprocessing import Manager, Process
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Mapping, Optional, Text

import numpy as np
import yaml
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.utils.helper import get_class_by_name
from tqdm import tqdm
from werkzeug.utils import secure_filename

from aTrain_core.globals import SAMPLING_RATE, TIMESTAMP_FORMAT
from aTrain_core.load_resources import get_model, load_model_config_file
from aTrain_core.outputs import (
    add_processing_time_to_metadata,
    assign_word_speakers,
    create_directory,
    create_file_id,
    create_metadata,
    create_output_files,
    named_tuple_to_dict,
    transform_speakers_results,
    write_logfile,
)
from aTrain_core.settings import Device, Settings
from aTrain_core.step_estimator import calculate_steps


class CustomPipeline(Pipeline):
    @classmethod
    def from_pretrained(cls, model_path) -> Pipeline:
        """Constructs a custom pipeline from pre-trained models."""
        config_yml = os.path.join(model_path, "config.yaml")
        with open(config_yml, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        params = config["pipeline"].get("params", {})
        path_segmentation_model = os.path.join(model_path, "segmentation_pyannote.bin")
        path_embedding_model = os.path.join(model_path, "embedding_pyannote.bin")
        params["segmentation"] = path_segmentation_model.replace("\\", "/")
        params["embedding"] = path_embedding_model.replace("\\", "/")
        pipeline: Pipeline = Klass(**params)
        pipeline.instantiate(config["params"])
        return pipeline


class CustomProgressHook(ProgressHook):
    """A custom progress hook that updates the GUI and prints progress information during processing."""

    def __init__(self, progress: DictProxy, total_steps):
        super().__init__()
        self._progress = progress
        self.completed_steps = 0
        self.total_steps = total_steps

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        super().__call__(step_name, step_artifact, file, total, completed)
        self._progress["task"] = "Detect Speakers"
        if self.step_name in ["speaker_counting", "discrete_diarization"]:
            self.completed_steps += 1
        if total and completed:
            self.completed_steps += 1
        self._progress["current"] = self.completed_steps
        self._progress["total"] = self.total_steps


def prepare_transcription(file: Path) -> tuple[Path, str, str]:
    """Create timestamp, file_id and directory for transcription"""

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    file = file.with_name(secure_filename(file.name))
    file_id = create_file_id(file, timestamp)
    create_directory(file_id)
    write_logfile(f"File ID created: {file_id}", file_id)
    return file, file_id, timestamp


def transcribe(settings: Settings):
    """Transcribes audio file with specified parameters."""

    write_logfile("Directory created", settings.file_id)
    audio_array, audio_duration = load_audio(settings)
    create_metadata(settings, audio_duration)
    model_path = get_model(settings.model)
    write_logfile("Model loaded", settings.file_id)
    if settings.device == Device.GPU:
        write_logfile("Transcribing in seperate process", settings.file_id)
        transcript = run_transcription_in_process(settings, model_path, audio_array)
    elif settings.device == Device.CPU:
        write_logfile("Transcribing in same process", settings.file_id)
        transcript = run_transcription(settings, model_path, audio_array)
    if settings.speaker_detection:
        transcript = run_speaker_detection(
            settings, audio_duration, audio_array, transcript
        )
    settings.progress["task"] = "Finish"
    create_output_files(transcript, settings.speaker_detection, settings.file_id)
    write_logfile("No speaker detection. Created output files", settings.file_id)
    add_processing_time_to_metadata(settings.file_id)
    write_logfile("Processing time added to metadata", settings.file_id)


def load_audio(settings: Settings) -> tuple[np.ndarray, int]:
    """Load the audio and calculate audio duration"""
    try:
        audio_array = decode_audio(settings.file, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        write_logfile(f"File or path invalid: {e}", settings.file_id)
        raise Exception("""Check file & path: File either has no audio or the name of the file path or file includes spaces. 
                        Please remove or exchange them with underscores.""")
    write_logfile("Audio file loaded and decoded", settings.file_id)
    audio_duration = int(len(audio_array) / SAMPLING_RATE)
    write_logfile("Audio duration calculated", settings.file_id)
    return audio_array, audio_duration


def run_transcription(
    settings: Settings,
    model_path: str,
    audio_array: np.ndarray,
    returnDict: DictProxy | dict = {},
) -> dict:
    """Run a transcription using a whisper model."""
    try:
        whisper_model = WhisperModel(
            model_size_or_path=model_path,
            device="cuda" if settings.device == Device.GPU else "cpu",
            compute_type=settings.compute_type.value,
        )
        model_type = load_model_config_file()[settings.model]["type"]
        write_logfile(f"Transcribing with {model_type} model.", settings.file_id)

        segments, info = whisper_model.transcribe(
            audio=audio_array,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
            language=None if settings.language == "auto-detect" else settings.language,
            max_new_tokens=None if model_type == "distil" else 128,
            no_speech_threshold=0.6,
            condition_on_previous_text=False if model_type == "distil" else True,
            initial_prompt=settings.initial_prompt,
        )
        segments = transcription_with_progress_bar(segments, info, settings.progress)
        transcript = {
            "segments": [named_tuple_to_dict(segment) for segment in segments]
        }
        write_logfile("Transcription successful", settings.file_id)
        if settings.device == Device.CPU:
            return transcript
        elif settings.device == Device.GPU:
            returnDict["transcript"] = transcript
            os._exit(0)

    except Exception as error:
        returnDict["error"] = error
        raise error


def transcription_with_progress_bar(transcription_segments, info, progress: DictProxy):
    """Transcribes audio segments with progress bar."""
    total_duration = round(info.duration, 2)
    timestamps = 0.0  # to get the current segments
    transcription_segments_new = []

    with tqdm(
        total=total_duration, unit=" audio seconds", desc="Transcribing with Whisper"
    ) as pbar:
        progress["task"] = "Transcribe"
        for segment in transcription_segments:
            transcription_segments_new.append(segment)
            progress["current"] = segment.end
            progress["total"] = total_duration
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration:  # silence at the end of the audio
            pbar.update(info.duration - timestamps)

    return transcription_segments_new


def run_transcription_in_process(
    settings: Settings, model_path: str, audio_array: np.ndarray
):
    """Run a transcription in a seperate process.
    This is a workaround to deal with a termination issue: https://github.com/guillaumekln/faster-whisper/issues/71"""
    with Manager() as manager:
        returnDict = manager.dict()
        p = Process(
            target=run_transcription,
            kwargs={
                "settings": settings,
                "model_path": model_path,
                "audio_array": audio_array,
                "returnDict": returnDict,
            },
            daemon=True,
        )
        p.start()
        p.join()
        p.close()

        if "error" in returnDict.keys():
            error: Exception = returnDict["error"]
            raise error
        else:
            transcript = returnDict["transcript"]
            return transcript


def run_speaker_detection(
    settings: Settings, audio_duration: int, audio_array: np.ndarray, transcript: dict
):
    """Run speaker detection using a pyannote.audio model"""
    import torch

    total_steps = calculate_steps(audio_duration)
    model_path = get_model("diarize")
    write_logfile("Speaker detection model loaded", settings.file_id)
    diarize_model = CustomPipeline.from_pretrained(model_path).to(torch.device("cpu"))
    write_logfile("Detecting speakers", settings.file_id)
    audio_array = {
        "waveform": torch.from_numpy(audio_array[None, :]),
        "sample_rate": SAMPLING_RATE,
    }
    with CustomProgressHook(settings.progress, total_steps) as hook:
        diarization_segments = diarize_model(
            audio_array,
            min_speakers=settings.speaker_count or None,
            max_speakers=settings.speaker_count or None,
            hook=hook,
        )
    speaker_results = transform_speakers_results(diarization_segments)
    write_logfile("Transformed diarization segments", settings.file_id)
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()
    transcript_with_speaker = assign_word_speakers(speaker_results, transcript)
    write_logfile("Assigned speakers to words", settings.file_id)
    return transcript_with_speaker


if __name__ == "__main__":
    ...
