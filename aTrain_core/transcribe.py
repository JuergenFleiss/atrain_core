import gc
import json
from importlib.resources import files

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from tqdm import tqdm

from .globals import MODELS_DIR, SAMPLING_RATE
from .GUI_integration import EventSender
from .load_resources import get_model
from .outputs import (
    add_processing_time_to_metadata,
    create_metadata,
    create_output_files,
    named_tuple_to_dict,
    transform_speakers_results,
    write_logfile,
)
from .step_estimator import calculate_steps


def transcription_with_progress_bar(transcription_segments, info, GUI: EventSender):
    """Transcribes audio segments with progress bar."""
    total_duration = round(info.duration, 2)
    timestamps = 0.0  # to get the current segments
    transcription_segments_new = []

    with tqdm(
        total=total_duration, unit=" audio seconds", desc="Transcribing with Whisper"
    ) as pbar:
        GUI.task_info("Transcribe")
        for nr, segment in enumerate(transcription_segments):
            transcription_segments_new.append(segment)
            GUI.progress_info(segment.end, total_duration)
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration:  # silence at the end of the audio
            pbar.update(info.duration - timestamps)

    return transcription_segments_new


def _prepare_metadata_creation(language, num_speakers, device, file_id, audio_file):
    """Preprocessing steps for the metadata creation and further use."""
    language = None if language == "auto-detect" else language
    min_speakers = max_speakers = (
        None if num_speakers == "auto-detect" else int(num_speakers)
    )
    device = "cuda" if device == "GPU" else "cpu"

    try:
        audio_array = decode_audio(audio_file, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        write_logfile(f"File has no audio: {e}", file_id)
        raise Exception("Attention: Your file has no audio.")
    write_logfile("Audio file loaded and decoded", file_id)
    audio_duration = int(len(audio_array) / SAMPLING_RATE)
    write_logfile("Audio duration calculated", file_id)
    return audio_array, audio_duration, device, min_speakers, max_speakers, language


def _perform_whisper_transcription(
    model_path,
    device,
    compute_type,
    audio_array,
    language,
    file_id,
    model,
    GUI: EventSender,
    initial_prompt=None,
):
    import torch

    transcription_model = WhisperModel(model_path, device, compute_type=compute_type)

    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)

    model_type = models[model]["type"]
    max_new_tokens = None if model_type == "distil" else 128

    write_logfile(f"Transcribing with {model_type} model.", file_id)

    transcription_segments, info = transcription_model.transcribe(
        audio=audio_array,
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
        language=language,
        max_new_tokens=max_new_tokens,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt
    )

    transcription_segments = transcription_with_progress_bar(
        transcription_segments, info, GUI
    )

    transcript = {
        "segments": [named_tuple_to_dict(segment) for segment in transcription_segments]
    }  # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
    write_logfile("Transcription successful", file_id)

    del transcription_model
    gc.collect()
    torch.cuda.empty_cache()
    return transcript


def _perform_pyannote_speaker_diarization(
    audio_duration,
    required_models_dir,
    file_id,
    GUI,
    min_speakers,
    max_speakers,
    audio_array,
    transcript,
):
    # We are importing this inside the function to speed up app start up.
    import torch
    from .pyannote_customizations import CustomPipeline, CustomProgressHook

    total_steps = calculate_steps(audio_duration)
    current_step = 0
    model_path = get_model("diarize", required_models_dir=required_models_dir)
    write_logfile("Speaker detection model loaded", file_id)
    diarize_model = CustomPipeline.from_pretrained(model_path, required_models_dir).to(
        torch.device("cpu")
    )
    write_logfile("Detecting speakers", file_id)
    audio_array = {
        "waveform": torch.from_numpy(audio_array[None, :]),
        "sample_rate": SAMPLING_RATE,
    }
    with CustomProgressHook(GUI, current_step, total_steps) as hook:
        diarization_segments = diarize_model(
            audio_array,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook,
        )

    if current_step < total_steps:
        current_step = total_steps
        GUI.progress_info(current_step, total_steps)
    speaker_results = transform_speakers_results(diarization_segments)
    write_logfile("Transformed diarization segments to speaker results", file_id)
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()
    transcript_with_speaker = _assign_word_speakers(speaker_results, transcript)
    write_logfile("Assigned speakers to words", file_id)
    return transcript_with_speaker


def _assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    """Assigns speakers to transcribed words."""
    # Function from whisperx -> see https://github.com/m-bain/whisperX.git
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        dia_tmp = (
            diarize_df[diarize_df["intersection"] > 0]
            if not fill_nearest
            else diarize_df
        )
        if len(dia_tmp) > 0:
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])
                    dia_tmp = (
                        diarize_df[diarize_df["intersection"] > 0]
                        if not fill_nearest
                        else diarize_df
                    )
                    if len(dia_tmp) > 0:
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker
    return transcript_result


def _finish_transcription_create_output_files(
    transcript, speaker_detection, file_id, GUI
):
    """Create output files after transcription."""
    GUI.task_info("Finish")
    create_output_files(transcript, speaker_detection, file_id)
    write_logfile("No speaker detection. Created output files", file_id)
    add_processing_time_to_metadata(file_id)
    write_logfile("Processing time added to metadata", file_id)


def transcribe(
    audio_file,
    file_id,
    model,
    language,
    speaker_detection,
    num_speakers,
    device,
    compute_type,
    timestamp,
    original_audio_filename,
    initial_prompt=None,
    GUI: EventSender = EventSender(),
    required_models_dir=MODELS_DIR,
):
    """Transcribes audio file with specified parameters."""
    # import inside function for faster startup times in GUI app

    GUI.task_info("Prepare")
    write_logfile("Directory created", file_id)

    audio_array, audio_duration, device, min_speakers, max_speakers, language = (
        _prepare_metadata_creation(language, num_speakers, device, file_id, audio_file)
    )
    create_metadata(
        file_id,
        file_id,
        audio_duration,
        model,
        language,
        speaker_detection,
        num_speakers,
        device,
        compute_type,
        timestamp,
        original_audio_filename,
    )
    write_logfile("Metadata created", file_id)

    if required_models_dir is None:
        required_models_dir = MODELS_DIR
    model_path = get_model(model, required_models_dir=required_models_dir)
    write_logfile("Model loaded", file_id)

    transcript = _perform_whisper_transcription(
        model_path, device, compute_type, audio_array, language, file_id, model, GUI, initial_prompt
    )

    if not speaker_detection:
        _finish_transcription_create_output_files(
            transcript, speaker_detection, file_id, GUI
        )

    if speaker_detection:
        transcript_with_speaker = _perform_pyannote_speaker_diarization(
            audio_duration,
            required_models_dir,
            file_id,
            GUI,
            min_speakers,
            max_speakers,
            audio_array,
            transcript,
        )
        _finish_transcription_create_output_files(
            transcript_with_speaker, speaker_detection, file_id, GUI
        )


if __name__ == "__main__":
    ...
