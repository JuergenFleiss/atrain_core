from transcribe import handle_transcription
from handle_upload import check_inputs, get_inputs, handle_file
from archive import read_archive, create_metadata, delete_transcription, open_file_directory, TIMESTAMP_FORMAT, ATRAIN_DIR
import traceback
from datetime import datetime
import argparse



def cli():
    parser = argparse.ArgumentParser(description="Process inputs for transcription.")
    parser.add_argument("file", help="Path to the audio file")
    parser.add_argument("--model", default="medium", help="Model to use for transcription")
    parser.add_argument("--language", default="auto-detect", help="Language of the audio")
    parser.add_argument("--speaker_detection", default=False, type=bool, help="Enable speaker detection (True/False)")
    parser.add_argument("--num_speakers", default="auto-detect", help="Number of speakers")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Device to use (CPU/GPU)")
    parser.add_argument("--compute_type", default="int8", choices=["float16", "int8"], help="Compute type (float16/int8)")
    
    args = parser.parse_args()
    print(args)

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    try:
        file, model, language, speaker_detection, num_speakers, device, compute_type = get_inputs(args) 
        inputs_correct = check_inputs(file, model, language, num_speakers)
        if inputs_correct is False:
            print("Incorrect inputs")
        filename, file_id, estimated_process_time, audio_duration = handle_file(file, timestamp, model, device)
        create_metadata(file_id, filename, audio_duration, model, language, speaker_detection, num_speakers, device, compute_type, timestamp)
        print(f"Transcription started for file {filename} with id {file_id} on device {device}. Estimated duration: {estimated_process_time} seconds")
        handle_transcription(file_id)
    except Exception as error:
        delete_transcription(timestamp)
        traceback_str = traceback.format_exc()
        error = str(error)
        print(f"An error has occurred: {traceback_str}")


if __name__ == "__main__":
    cli()