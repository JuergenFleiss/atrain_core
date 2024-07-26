from .outputs import create_output_files, named_tuple_to_dict, transform_speakers_results, create_directory, add_processing_time_to_metadata, create_metadata, write_logfile
from .globals import SAMPLING_RATE, MODELS_DIR, audio_lengths, embedding_steps, segmentation_steps
from .load_resources import get_model
from .GUI_integration import EventSender
from faster_whisper.audio import decode_audio
import numpy as np
import gc, torch #Import inside the function to speed up the startup time of the destkop app.
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionOptions, Segment
from faster_whisper.audio import decode_audio
from faster_whisper.tokenizer import Tokenizer
from pyannote.audio import Pipeline
from pyannote.core.utils.helper import get_class_by_name
from pyannote.audio.pipelines.utils.hook import ProgressHook
from importlib.resources import files
import yaml
import json
import math
import os
import time
from tqdm import tqdm
from typing import Mapping, Optional, Iterable, Optional, Text, Any
from .step_estimator import QuadraticRegressionModel
import ctranslate2




class CustomPipeline(Pipeline):
    @classmethod
    def from_pretrained(cls,model_path) -> "Pipeline":
        """Constructs a custom pipeline from pre-trained models."""
        config_yml = os.path.join(MODELS_DIR, "diarize", "config.yaml")
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
    """A custom progress hook that updates the GUI and prints progress information during processing."""
    def __init__(self, GUI: EventSender, completed_steps, total_steps):
        super().__init__()
        self.GUI = GUI
        self.completed_steps = completed_steps
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

        self.GUI.task_info(f"{self.step_name}")
        if self.step_name == "speaker_counting" or self.step_name == "discrete_diarization":
            self.completed_steps += 1
            self.GUI.progress_info(self.completed_steps, self.total_steps)
        
        if total is not None and completed is not None:
            if completed != 0:
                self.completed_steps += 1
                self.GUI.progress_info(self.completed_steps, self.total_steps)
        
        
                
                
                
                

            
def transcription_with_progress_bar(transcription_segments, info, GUI : EventSender, completed_steps, total_steps):
    """Transcribes audio segments with progress bar."""
    total_duration = round(info.duration, 2)  
    timestamps = 0.0  # to get the current segments
    transcription_segments_new = []

    

    with tqdm(total=total_duration, unit=" audio seconds", desc="Transcribing with Whisper") as pbar:
        GUI.task_info("transcribing with Whisper model")
        for nr, segment in enumerate(transcription_segments):
            completed_steps +=1
            print(f"Progress: {completed_steps}/{total_steps}")
            GUI.progress_info(completed_steps, total_steps)
        
            
            transcription_segments_new.append(segment)
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration: # silence at the end of the audio
            pbar.update(info.duration - timestamps)

    
    return transcription_segments_new
            

class CountingWhisperModel(WhisperModel):
    """A subclass of WhisperModel that counts the total number of generated segments during transcription."""
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

        # Generating segments for the counter
        segments = super().generate_segments(features, tokenizer, options, encoder_output)

        # Count segments
        self.total_segments = sum(1 for _ in segments)

        # Generating segments again for further processing
        segments = super().generate_segments(features, tokenizer, options, encoder_output)

        return segments
    

def calculate_steps(speaker_detection, nr_segments, audio_duration):
    """Calculates the total number of steps for the transcription process."""
    # Initialize model
    model = QuadraticRegressionModel()

    # Train the model
    model.train(audio_lengths, segmentation_steps, embedding_steps)

    total_steps = 0
    if not speaker_detection:
        total_steps = nr_segments
        print(f"Total steps without diarization: {total_steps}")
        return total_steps

    elif speaker_detection:
        total_steps += nr_segments
        # Predictions
        segmentation_prediction = model.predict_segmentation(audio_duration)
        embedding_prediction = model.predict_embedding(audio_duration)


        segmentation_prediction = segmentation_prediction/32
        # account for one extra process when not divisible by 32
        if isinstance(segmentation_prediction/32, int):
            segmentation_prediction = segmentation_prediction
        else:
            segmentation_prediction = math.ceil(segmentation_prediction+1)

        total_steps += segmentation_prediction + embedding_prediction + 2 # speaker_counting & discrete_diarization are one step each
        return total_steps
    


    

def transcribe(audio_file, file_id, model, language, speaker_detection, num_speakers, device, compute_type, timestamp, GUI : EventSender = EventSender()):
    """Transcribes audio file with specified parameters.""" 
    start_time = time.time()
    num_steps = 0
    GUI.task_info("preparing transcription")
    print("Preparing transcription")
    GUI.progress_info(num_steps, 100) # we cannot calculate the total number of steps yet, so default is 100
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
    
    else:
        write_logfile("Transcribing with regular multilingual model", file_id)
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language,max_new_tokens=128, no_speech_threshold=0.6, condition_on_previous_text=False)

    
    total_steps = math.ceil(calculate_steps(speaker_detection, transcription_model.total_segments, audio_duration))
    GUI.progress_info(num_steps, total_steps)
    
    transcription_segments = transcription_with_progress_bar(transcription_segments, info, GUI, num_steps, total_steps)
    transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]} # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
    num_steps = transcription_model.total_segments
    write_logfile("Transcription successful", file_id) 


    del transcription_model; gc.collect(); torch.cuda.empty_cache()
    
    if not speaker_detection:
        print(f"Finishing up")
        GUI.task_info("Creating output files")
        create_output_files(transcript, speaker_detection, file_id)
        write_logfile("No speaker detection. Created output files", file_id)
        add_processing_time_to_metadata(file_id)
        write_logfile("Processing time added to metadata", file_id)
        GUI.finished_info(file_id)

    if speaker_detection:
        print("Loading speaker detection model")
        model_path = get_model("diarize")
        write_logfile("Speaker detection model loaded", file_id)
        diarize_model = CustomPipeline.from_pretrained(model_path).to(torch.device("cpu"))
        write_logfile("Detecting speakers", file_id)
        audio_array = { "waveform": torch.from_numpy(audio_array[None, :]), "sample_rate": SAMPLING_RATE}
        with CustomProgressHook(GUI, num_steps, total_steps) as hook:
            diarization_segments = diarize_model(audio_array,min_speakers=min_speakers, max_speakers=max_speakers, hook=hook)
        
        if num_steps < total_steps:
            num_steps = total_steps
            GUI.progress_info(num_steps, total_steps)
            print(f"Progress {num_steps}/{total_steps}")
        speaker_results = transform_speakers_results(diarization_segments)
        write_logfile("Transformed diarization segments to speaker results", file_id)
        del diarize_model; gc.collect(); torch.cuda.empty_cache()
        transcript_with_speaker = assign_word_speakers(speaker_results,transcript)
        write_logfile("Assigned speakers to words", file_id)
        print("Finishing up")
        GUI.task_info("Creating output files")
        create_output_files(transcript_with_speaker, speaker_detection, file_id)
        write_logfile("Created output files", file_id)
        add_processing_time_to_metadata(file_id)
        write_logfile("Processing time added to metadata", file_id)
        GUI.finished_info(file_id)

def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    """Assigns speakers to transcribed words."""
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
    
