from .outputs import OutputHandler, named_tuple_to_dict, transform_speakers_results
from .globals import SAMPLING_RATE, ATRAIN_DIR
from .load_resources import get_model
from faster_whisper.audio import decode_audio
import numpy as np
import gc, torch #Import inside the function to speed up the startup time of the destkop app.
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core.utils.helper import get_class_by_name
from importlib.resources import files
import yaml
import json
import os
from tqdm import tqdm
from pyannote.audio.pipelines.utils.hook import ProgressHook

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
        for segment in transcription_segments:
            transcription_segments_new.append(segment)
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration: # silence at the end of the audio
            pbar.update(info.duration - timestamps)
        
    return transcription_segments_new
            

    

def transcribe (audio_file, output_handler, model, language, speaker_detection, num_speakers, device, compute_type, timestamp):
    """Transcribes audio file with specified parameters.

    Args:
        audio_file (str): Path to the audio file.
        output_handler (OutputHandler): Identifier for the file.
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
    output_handler.create_directory()
    output_handler.write_logfile("Directory created")
    language = None if language == "auto-detect" else language
    min_speakers = max_speakers = None if num_speakers == "auto-detect" else int(num_speakers)
    device = "cuda" if device=="GPU" else "cpu"

    audio_array = decode_audio(audio_file, sampling_rate=SAMPLING_RATE)
    output_handler.write_logfile("Audio file loaded and decoded")
    audio_duration = int(len(audio_array)/SAMPLING_RATE)
    output_handler.write_logfile("Audio duration calculated")
    output_handler.create_metadata(audio_duration, model, language, speaker_detection, num_speakers, device, compute_type, timestamp)
    output_handler.write_logfile("Metadata created")

    model_path = get_model(model)
    output_handler.write_logfile("Model loaded")
    transcription_model = WhisperModel(model_path,device,compute_type=compute_type)

    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)

    if models[model]["type"] == "distil":
        output_handler.write_logfile("Transcribing with distil model")
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language, no_speech_threshold=0.6, condition_on_previous_text=False)
        transcription_segments = transcription_with_progress_bar(transcription_segments, info)
        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]} # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
        output_handler.write_logfile("Transcription successful")

    else:
        output_handler.write_logfile("Transcribing with regular multilingual model")
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language,max_new_tokens=128, no_speech_threshold=0.6, condition_on_previous_text=False)
        transcription_segments = transcription_with_progress_bar(transcription_segments, info)
        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]} # wenn man die beiden umdreht also progress bar zuerst damit er schön läuft, dann ist das segments dict leer, sprich es gibt keine transkription
        output_handler.write_logfile("Transcription successful")


    del transcription_model; gc.collect(); torch.cuda.empty_cache()
    
    if not speaker_detection:
        print(f"Finishing up")
        output_handler.create_output_files(transcript, speaker_detection)
        output_handler.write_logfile("No speaker detection. Created output files")
        output_handler.add_processing_time_to_metadata()
        output_handler.write_logfile("Processing time added to metadata")

    if speaker_detection:
        print("Loading speaker detection model")
        model_path = get_model("diarize")
        output_handler.write_logfile("Speaker detection model loaded")
        diarize_model = CustomPipeline.from_pretrained(model_path).to(torch.device("cpu"))
        output_handler.write_logfile("Detecting speakers")
        audio_array = { "waveform": torch.from_numpy(audio_array[None, :]), "sample_rate": SAMPLING_RATE}
        with ProgressHook() as hook:
            diarization_segments = diarize_model(audio_array,min_speakers=min_speakers, max_speakers=max_speakers, hook=hook)
        speaker_results = transform_speakers_results(diarization_segments)
        output_handler.write_logfile("Transformed diarization segments to speaker results")
        del diarize_model; gc.collect(); torch.cuda.empty_cache()
        transcript_with_speaker = assign_word_speakers(speaker_results,transcript)
        output_handler.write_logfile("Assigned speakers to words")
        print("Finishing up")
        output_handler.create_output_files(transcript_with_speaker, speaker_detection)
        output_handler.write_logfile("Created output files")
        output_handler.add_processing_time_to_metadata()
        output_handler.write_logfile("Processing time added to metadata")

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
    
