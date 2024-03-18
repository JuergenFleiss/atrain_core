from transcribe import handle_transcription
from handle_upload import check_inputs, get_inputs, handle_file
from archive import read_archive, create_metadata, delete_transcription, open_file_directory, TIMESTAMP_FORMAT, ATRAIN_DIR
import traceback
from datetime import datetime
import argparse
from load_resources import download_all_resources



def cli():
    parser = argparse.ArgumentParser(prog='aTrain_core', description='A CLI tool for audio transcription with Whisper')
    subparsers = parser.add_subparsers(dest='command', help='Command for aTrain_core to perform.')

    # Subparser for 'init' command
    parser_init = subparsers.add_parser('init', help='Initialize aTrain_core by downloading models')

    # Subparser for 'transcribe' command
    parser_start = subparsers.add_parser('start', help='Start transcription process for an audio file')
    parser_start.add_argument('file', help='Path to the audio file')
    parser_start.add_argument("--model", default="medium", help="Model to use for transcription")
    parser_start.add_argument("--language", default="auto-detect", help="Language of the audio")
    parser_start.add_argument("--speaker_detection", default=False, action="store_true", help="Enable speaker detection")
    parser_start.add_argument("--num_speakers", default="auto-detect", help="Number of speakers")
    parser_start.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Device to use (CPU/GPU)")
    parser_start.add_argument("--compute_type", default="int8", choices=["float16", "int8"], help="Compute type (float16/int8)")

    args = parser.parse_args()

    if args.command == "init":
        print("Downloading all models:")
        download_all_resources()
        print("Finished")

    elif args.command == "start":
        print("Running aTrain_core")
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        try:
            file, model, language, speaker_detection, num_speakers, device, compute_type = args.file, args.model, args.language, args.speaker_detection, args.num_speakers, args.device, args.compute_type
            inputs_correct = check_inputs(file, model, language, num_speakers)
            if not inputs_correct:
                print("Incorrect inputs")
            else:
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





# THIS IS THE CLI FUNCTION FROM THE ORIGINAL aTrain/__init__.py

# def cli():
#     parser = argparse.ArgumentParser(prog='aTrain_core', description='A GUI tool to transcribe audio with Whisper')
#     parser.add_argument("command", choices=['init', 'start'], help="Command for aTrain_core to perform.")
#     args = parser.parse_args()

#     if args.command == "init":
#         print("Downloading all models:")
#         download_all_resources()
#         print("Finished")
#     if args.command == "start":
#         print("Running aTrain_core")
#         run_app()