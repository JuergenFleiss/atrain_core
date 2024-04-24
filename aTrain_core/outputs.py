import os
import json 
import pandas as pd
import time
import shutil
import yaml
from datetime import datetime
from .globals import TRANSCRIPT_DIR, METADATA_FILENAME, TIMESTAMP_FORMAT, LOG_FILENAME



def create_directory(file_id):
    """Creates a directory for storing transcription files.

    Args:
        file_id (str): Identifier for the file.

    Returns:
        None
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    file_directory = os.path.join(TRANSCRIPT_DIR, file_id)
    os.makedirs(file_directory, exist_ok=True)
    print(f"Created directory at {file_directory}")
    


def create_file_id(file_path, timestamp):
    """Creates a unique identifier for a file composed of the file path and timestamp.

    Args:
        file_path (str): Path to the file.
        timestamp (str): Timestamp for the file.

    Returns:
        str: Unique identifier for the file.
    """
      # Extract filename from file_path
    file_base_name = os.path.basename(file_path)
    short_base_name = file_base_name[0:5] if len(file_base_name) >= 5 else file_base_name
    file_id = timestamp + " " + short_base_name
    return file_id


def create_output_files(result, speaker_detection, file_id):
    """Creates output files based on the transcription result.

    Args:
        result (dict): Transcription result.
        speaker_detection (bool): Whether speaker detection was performed.
        file_id (str): Identifier for the file.

    Returns:
        None
    """
    create_json_file(result, file_id)
    create_txt_file(result, file_id, speaker_detection, maxqda=False, timestamps = False)
    create_txt_file(result, file_id,speaker_detection, maxqda=False, timestamps = True)
    create_txt_file(result, file_id, speaker_detection, maxqda=True, timestamps = True)
    create_srt_file(result, file_id)

def create_json_file(result, file_id):
    """Creates a JSON file for the transcription result.

    Args:
        result (dict): Transcription result.
        file_id (str): Identifier for the file.

    Returns:
        None
    """
    output_file_text = os.path.join(TRANSCRIPT_DIR,file_id,"transcription.json")
    with open(output_file_text,"w", encoding="utf-8") as json_file:
        json.dump(result, json_file,ensure_ascii=False)

def create_txt_file (result,file_id, speaker_detection, timestamps, maxqda):
    """Creates a TXT file for the transcription result.

    Args:
        result (dict): Transcription result.
        file_id (str): Identifier for the file.
        speaker_detection (bool): Whether speaker detection was performed.
        timestamps (bool): Whether to include timestamps.
        maxqda (bool): Whether to format for MaxQDA.

    Returns:
        None
    """
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
    """Creates a SRT file for the transcription result.

    Args:
        result (dict): Transcription result.
        file_id (str): Identifier for the file.

    Returns:
        None
    """

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
    """Transforms diarization segments to speaker results.

    Args:
        diarization_segments (list): Diarization segments.

    Returns:
        DataFrame: Dataframe containing speaker information.
    """

    diarize_df = pd.DataFrame(diarization_segments.itertracks(yield_label=True))
    diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
    diarize_df.rename(columns={2: "speaker"}, inplace=True)
    return diarize_df

def named_tuple_to_dict(obj):
    """Converts named tuple to dictionary.

    Args:
        obj (object): Object to convert.

    Returns:
        dict: Converted dictionary.
    """
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
    """Checks if the object is an instance of namedtuple.

    Args:
        x (object): Object to check.

    Returns:
        bool: True if the object is an instance of namedtuple, False otherwise.
    """
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i)==str for i in fields)


def create_metadata(file_id, filename, audio_duration, model, language, speaker_detection, num_speakers, device, compute_type, timestamp):
    """Creates metadata file for the transcription.

    Args:
        file_id (str): Identifier for the file.
        filename (str): Name of the file.
        audio_duration (int): Duration of the audio.
        model (str): Name of the transcription model.
        language (str): Language used for transcription.
        speaker_detection (bool): Whether speaker detection was performed.
        num_speakers (int): Number of speakers detected.
        device (str): Device used for transcription.
        compute_type (str): Type of compute used.
        timestamp (str): Timestamp for the transcription.

    Returns:
        None
    """

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

def write_logfile(message, file_id):
    """Writes a log message to the log file.

    Args:
        message (str): Log message.
        file_id (str): Identifier for the file.

    Returns:
        None
    """

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    log_file_path = os.path.join(TRANSCRIPT_DIR,file_id,LOG_FILENAME)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ------ {message}\n")


def add_processing_time_to_metadata(file_id):
    """Adds processing time information to metadata.

    Args:
        file_id (str): Identifier for the file.

    Returns:
        None
    """

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
    """Deletes the transcription files.

    Args:
        file_id (str): Identifier for the file.

    Returns:
        None
    """
    
    file_id = "" if file_id == "all" else file_id
    directory_name = os.path.join(TRANSCRIPT_DIR,file_id)
    if os.path.exists(directory_name):    
        shutil.rmtree(directory_name)
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)