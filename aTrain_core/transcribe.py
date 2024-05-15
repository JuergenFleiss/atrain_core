from .outputs import create_output_files, named_tuple_to_dict, transform_speakers_results, create_directory, add_processing_time_to_metadata, create_metadata, write_logfile
from .globals import SAMPLING_RATE, ATRAIN_DIR, audio_lengths, embedding_steps, segmentation_steps
from .load_resources import get_model
from .GUI_integration import EventSender
from faster_whisper.audio import decode_audio
import numpy as np
import gc, torch #Import inside the function to speed up the startup time of the destkop app.
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionOptions, Segment
from pyannote.audio import Pipeline
from pyannote.core.utils.helper import get_class_by_name
from importlib.resources import files
import yaml
import json
import os
from tqdm import tqdm
from pyannote.audio.pipelines.utils.hook import ProgressHook, Hooks, TimingHook, ArtifactHook
from typing import Any, Mapping, Optional, Text
import json
import os
from .step_estimator import QuadraticRegressionModel


from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union

import ctranslate2
import numpy as np

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer


class CustomPipeline(Pipeline):
    @classmethod
    def from_pretrained(cls,model_path) -> "Pipeline":
        """Constructs a custom pipeline from pre-trained models.

        Args:
            model_path (str): Path to the directory containing pre-trained models.

        Returns:
            Pipeline: An instance of the custom pipeline configured with the pre-trained models.
        """
        config_yml = os.path.join(ATRAIN_DIR, "models", "diarize", "config.yaml")
        with open(config_yml, "r") as config_file: 
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(pipeline_name, default_module_name="pyannote.pipeline.blocks")
        params = config["pipeline"].get("params", {})
        path_segmentation_model = os.path.join(model_path,"segmentation_pyannote.bin")
        path_embedding_model = os.path.join(model_path,"embedding_pyannote.bin")
        params["segmentation"] = path_segmentation_model.replace('\\', '/')
        print(params["segmentation"])
        params["embedding"] = path_embedding_model.replace('\\', '/')
        print(params["embedding"])
        pipeline = Klass(**params)
        pipeline.instantiate(config["params"])
        return pipeline
    
class CustomProgressHook(ProgressHook):
    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        super().__call__(step_name, step_artifact, file, total, completed)

        # Print the current step and progress
        print(f"Current step: {self.step_name}")
        if total is not None and completed is not None:
            print(f"Progress: {completed}/{total}")

            
def transcription_with_progress_bar(transcription_segments, info):
    """Transcribes audio segments with progress bar.

    Args:
        transcription_segments (list): List of audio segments to transcribe.
        info (object): Information about the audio.

    Returns:
        list: Transcribed audio segments with progress bar.
    """
    total_duration = round(info.duration, 2)  
    timestamps = 0.0  # to get the current segments
    transcription_segments_new = []

    

    with tqdm(total=total_duration, unit=" audio seconds", desc="Transcribing with Whisper") as pbar:
        for nr, segment in enumerate(transcription_segments):
            print(f"Segment {nr}: {segment.text}")
            transcription_segments_new.append(segment)
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration: # silence at the end of the audio
            pbar.update(info.duration - timestamps)
        
    print(len(transcription_segments_new))
    return transcription_segments_new
            

class CountingWhisperModel(WhisperModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_segments = 0

    def generate_segments(
            self,
            features: np.ndarray,
            tokenizer: Tokenizer,
            options: TranscriptionOptions,
            encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        # Reset total segments counter
        self.total_segments = 0

        # Your existing code for generating segments
        segments = super().generate_segments(features, tokenizer, options, encoder_output)

        # Count segments
        self.total_segments = sum(1 for _ in segments)

        # Return the generator again for iteration
        segments = super().generate_segments(features, tokenizer, options, encoder_output)

        return segments
    

def calculate_steps(speaker_detection, nr_segments, audio_duration):
    # Initialize model
    model = QuadraticRegressionModel()

    # Train the model
    model.train(audio_lengths, segmentation_steps, embedding_steps)

    total_steps = 0
    if not speaker_detection:
        total_steps == nr_segments
        print(f"Total steps without diarization: {total_steps}")

    elif speaker_detection:
        total_steps += nr_segments
        # Predictions
        segmentation_prediction = model.predict_segmentation(audio_duration)
        embedding_prediction = model.predict_embedding(audio_duration)
        total_steps += segmentation_prediction + embedding_prediction + 2 # speaker_counting & discrete_diarization are one step each

        print(f"Segmentation Prediction for length {audio_duration}: {segmentation_prediction:.0f}")
        print(f"Embedding Prediction for length {audio_duration}: {embedding_prediction:.0f}")
        print(f"(Predicted) total steps with diarization: {total_steps:.0f}")

    

def transcribe(audio_file, file_id, model, language, speaker_detection, num_speakers, device, compute_type, timestamp, GUI : EventSender = EventSender()):
    """Transcribes audio file with specified parameters.

    Args:
        audio_file (str): Path to the audio file.
        file_id (str): Identifier for the file.
        model (str): Name of the transcription model.
        language (str): Language for transcription.
        speaker_detection (bool): Whether to perform speaker detection.
        num_speakers (str): Number of speakers for speaker detection.
        device (str): Device to use for transcription.
        compute_type (str): Type of compute to use.
        timestamp (str): Timestamp for the transcription.

    Returns:
        None
    """ 

    number_of_segments = 0
    #Send data to the GUI e.g., like so -> GUI.task_info("current task")
    create_directory(file_id)
    write_logfile("Directory created", file_id)
    language = None if language == "auto-detect" else language
    min_speakers = max_speakers = None if num_speakers == "auto-detect" else int(num_speakers)
    device = "cuda" if device=="GPU" else "cpu"

    audio_array = decode_audio(audio_file, sampling_rate=SAMPLING_RATE)
    write_logfile("Audio file loaded and decoded", file_id)
    audio_duration = int(len(audio_array)/SAMPLING_RATE)
    write_logfile("Audio duration calculated", file_id)
    create_metadata(file_id, file_id, audio_duration, model, language, speaker_detection, num_speakers, device, compute_type, timestamp)
    write_logfile("Metadata created", file_id)

    model_path = get_model(model)
    write_logfile("Model loaded", file_id)
    transcription_model = CountingWhisperModel(model_path,device,compute_type=compute_type)
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)

    if models[model]["type"] == "distil":
        write_logfile("Transcribing with distil model", file_id)
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language, no_speech_threshold=0.6, condition_on_previous_text=False)
        calculate_steps(speaker_detection, transcription_model.total_segments, audio_duration)
        transcription_segments = transcription_with_progress_bar(transcription_segments, info)
        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]} # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
        write_logfile("Transcription successful", file_id)

    else:
        write_logfile("Transcribing with regular multilingual model", file_id)
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language,max_new_tokens=128, no_speech_threshold=0.6, condition_on_previous_text=False)
        calculate_steps(speaker_detection, transcription_model.total_segments, audio_duration)
        transcription_segments = transcription_with_progress_bar(transcription_segments, info)
        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]} # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
        write_logfile("Transcription successful", file_id)


    del transcription_model; gc.collect(); torch.cuda.empty_cache()
    
    if not speaker_detection:
        print(f"Finishing up")
        create_output_files(transcript, speaker_detection, file_id)
        write_logfile("No speaker detection. Created output files", file_id)
        add_processing_time_to_metadata(file_id)
        write_logfile("Processing time added to metadata", file_id)

    if speaker_detection:
        print("Loading speaker detection model")
        model_path = get_model("diarize")
        write_logfile("Speaker detection model loaded", file_id)
        diarize_model = CustomPipeline.from_pretrained(model_path).to(torch.device("cpu"))
        write_logfile("Detecting speakers", file_id)
        audio_array = { "waveform": torch.from_numpy(audio_array[None, :]), "sample_rate": SAMPLING_RATE}
        with CustomProgressHook() as hook:
            diarization_segments = diarize_model(audio_array,min_speakers=min_speakers, max_speakers=max_speakers, hook=hook)
           
        speaker_results = transform_speakers_results(diarization_segments)
        write_logfile("Transformed diarization segments to speaker results", file_id)
        del diarize_model; gc.collect(); torch.cuda.empty_cache()
        transcript_with_speaker = assign_word_speakers(speaker_results,transcript)
        write_logfile("Assigned speakers to words", file_id)
        print("Finishing up")
        create_output_files(transcript_with_speaker, speaker_detection, file_id)
        write_logfile("Created output files", file_id)
        add_processing_time_to_metadata(file_id)
        write_logfile("Processing time added to metadata", file_id)

def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    """Assigns speakers to transcribed words.

    Args:
        diarize_df (DataFrame): Dataframe containing speaker information.
        transcript_result (dict): Transcription result.
        fill_nearest (bool, optional): Whether to fill nearest speakers. Defaults to False.

    Returns:
        dict: Transcription result with assigned speakers.
    """
    #Function from whisperx -> see https://github.com/m-bain/whisperX.git
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        dia_tmp = diarize_df[diarize_df['intersection'] > 0] if not fill_nearest else diarize_df
        if len(dia_tmp) > 0:
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0] if not fill_nearest else diarize_df
                    if len(dia_tmp) > 0:
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker
    return transcript_result    

if __name__ == "__main__":
    ...
    
