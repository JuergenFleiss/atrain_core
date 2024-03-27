import os
import json 
import pandas as pd
import time
import shutil
import yaml
from datetime import datetime
from .globals import TRANSCRIPT_DIR, METADATA_FILENAME, TIMESTAMP_FORMAT



def create_directory(file_id):

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    file_directory = os.path.join(TRANSCRIPT_DIR, file_id)
    os.makedirs(file_directory, exist_ok=True)
    print(f"Created directory at {file_directory}")
    


def create_file_id(file_path, timestamp):
      # Extract filename from file_path
    file_base_name = os.path.basename(file_path)
    short_base_name = file_base_name[0:49] if len(file_base_name) >= 50 else file_base_name
    file_id = timestamp + " " + short_base_name
    return file_id


def create_output_files(result, speaker_detection, file_id):
        create_json_file(result, file_id)
        create_txt_file(result, file_id, speaker_detection, maxqda=False, timestamps = False)
        create_txt_file(result, file_id,speaker_detection, maxqda=False, timestamps = True)
        create_txt_file(result, file_id, speaker_detection, maxqda=True, timestamps = True)
        create_srt_file(result, file_id)

def create_json_file(result, file_id):
        output_file_text = os.path.join(TRANSCRIPT_DIR,file_id,"transcription.json")
        with open(output_file_text,"w", encoding="utf-8") as json_file:
            json.dump(result, json_file,ensure_ascii=False)

def create_txt_file (result,file_id, speaker_detection, timestamps, maxqda):
    segments = result["segments"]
    match maxqda, timestamps:
         case True, _ :  filename = "transcription_maxqda.txt"
         case False, True: filename = "transcription_timestamps.txt"
         case False, False: filename = "transcription.txt"
    file_path = os.path.join(TRANSCRIPT_DIR,file_id, filename)
    with open(file_path, "w",encoding="utf-8") as file:
        headline = f"Transcription for {file_id}" + ( "" if maxqda and speaker_detection else "\n") + ("" if speaker_detection else "\n" )
        file.write(headline)
        current_speaker = None
        for segment in segments:
            speaker = segment["speaker"] if "speaker" in segment else "Speaker undefined"
            if speaker != current_speaker and speaker_detection:
                file.write(("\n\n" if maxqda else "\n") + speaker + "\n")
                current_speaker = speaker
            text = str(segment["text"]).lstrip()
            if timestamps:
                start_time = time.strftime("[%H:%M:%S]", time.gmtime(segment["start"]))
                text = f"{start_time} - {text}"
            file.write(text + (" " if maxqda else  "\n"))

def create_srt_file(result,file_id):
    segments = result["segments"]
    file_path = os.path.join(TRANSCRIPT_DIR,file_id, "transcription.srt")
    with open(file_path,"w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments,1):
            srt_file.write(f"{index}\n")
            start_time = segment["start"]
            end_time = segment["end"]
            start_time_format = time.strftime("%H:%M:%S", time.gmtime(start_time)) + f",{round((start_time-int(start_time))*1000):03}"
            end_time_format = time.strftime("%H:%M:%S", time.gmtime(end_time)) + f",{round((end_time-int(end_time))*1000):03}"
            srt_file.write(f"{start_time_format} --> {end_time_format}\n")
            srt_file.write(f"{str(segment['text']).lstrip()}\n\n")

def transform_speakers_results(diarization_segments):    
    diarize_df = pd.DataFrame(diarization_segments.itertracks(yield_label=True))
    diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
    diarize_df.rename(columns={2: "speaker"}, inplace=True)
    return diarize_df

def named_tuple_to_dict(obj):
    if isinstance(obj, dict):
        return {key: named_tuple_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [named_tuple_to_dict(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: named_tuple_to_dict(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(named_tuple_to_dict(value) for value in obj)
    else:
        return obj

def isnamedtupleinstance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i)==str for i in fields)


def create_metadata(file_id, filename, audio_duration, model, language, speaker_detection, num_speakers, device, compute_type, timestamp):
    metadata_file_path = os.path.join(TRANSCRIPT_DIR,file_id,METADATA_FILENAME)
    metadata = {
        "file_id" : file_id,
        "filename" : filename,
        "audio_duration" : audio_duration,
        "model" : model,
        "language" : language,
        "speaker_detection" : speaker_detection,
        "num_speakers" : num_speakers,
        "device": device,
        "compute_type" : compute_type,
        "timestamp": timestamp 
        }
    with open(metadata_file_path,"w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata, metadata_file)


def add_processing_time_to_metadata(file_id):
    metadata_file_path = os.path.join(TRANSCRIPT_DIR,file_id,METADATA_FILENAME)
    with open(metadata_file_path, "r", encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    timestamp = metadata["timestamp"]
    start_time = datetime.strptime(timestamp,TIMESTAMP_FORMAT)
    stop_time = timestamp = datetime.now()
    processing_time = stop_time-start_time
    metadata["processing_time"] = int(processing_time.total_seconds())
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata,metadata_file)

def delete_transcription(file_id):
    file_id = "" if file_id == "all" else file_id
    directory_name = os.path.join(TRANSCRIPT_DIR,file_id)
    if os.path.exists(directory_name):    
        shutil.rmtree(directory_name)
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)