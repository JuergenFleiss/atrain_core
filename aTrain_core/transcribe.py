from .outputs import create_output_files, named_tuple_to_dict, transform_speakers_results, create_directory, add_processing_time_to_metadata, create_metadata, write_logfile
from .globals import SAMPLING_RATE
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
        config_yml = str(files("aTrain_core.models").joinpath("config.yaml"))
        with open(config_yml, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(pipeline_name, default_module_name="pyannote.pipeline.blocks")
        params = config["pipeline"].get("params", {})
        path_segmentation_model = os.path.join(model_path,"segmentation_pyannote.bin")
        path_embedding_model = os.path.join(model_path,"embedding_pyannote.bin")
        params["segmentation"] = path_segmentation_model.replace('\\', '/')
        params["embedding"] = path_embedding_model.replace('\\', '/')
        pipeline = Klass(**params)
        pipeline.instantiate(config["params"])
        return pipeline

def transcription_progress_bar(transcription_segments, info):
    total_duration = round(info.duration, 2)  
    timestamps = 0.0  # to get the current segments
    with tqdm(total=total_duration, unit=" audio seconds", desc="Transcribing file with whisper") as pbar:
        for segment in transcription_segments:
            pbar.update(segment.end - timestamps)
            timestamps = segment.end
        if timestamps < info.duration: # silence at the end of the audio
            pbar.update(info.duration - timestamps)


def transcribe (audio_file, file_id, model, language, speaker_detection, num_speakers, device, compute_type, timestamp):   
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
    transcription_model = WhisperModel(model_path,device,compute_type=compute_type)

    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)

    if models[model]["type"] == "distil":
        write_logfile("Transcribing with distil model", file_id)
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language, no_speech_threshold=0.6, condition_on_previous_text=False)
        transcription_progress_bar(transcription_segments, info)

        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]}
        write_logfile("Transcription successful", file_id)

    else:
        write_logfile("Transcribing with regular multilingual model", file_id)
        transcription_segments, info = transcription_model.transcribe(audio=audio_array,vad_filter=True, beam_size=5, word_timestamps=True,language=language,max_new_tokens=128, no_speech_threshold=0.6, condition_on_previous_text=False)
        transcription_progress_bar(transcription_segments, info) 
        transcript = {"segments":[named_tuple_to_dict(segment) for segment in transcription_segments]}
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
        diarize_model = CustomPipeline.from_pretrained(model_path).to(torch.device(device))
        write_logfile("Detecting speakers", file_id)
        audio_array = { "waveform": torch.from_numpy(audio_array[None, :]), "sample_rate": SAMPLING_RATE}
        with ProgressHook() as hook:
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
    