from .transcribe import transcribe
from .check_inputs import check_inputs_transcribe
from .outputs import create_file_id, delete_transcription
from .transcribe import transcribe
from .globals import TIMESTAMP_FORMAT
from .load_resources import download_all_resources
import traceback
from datetime import datetime
import argparse





def cli():
    parser = argparse.ArgumentParser(prog='aTrain_core', description='A CLI tool for audio transcription with Whisper')
    subparsers = parser.add_subparsers(dest='command', help='Command for aTrain_core to perform.')

    # Subparser for 'init' command
    parser_load = subparsers.add_parser('load', help='Initialize aTrain_core by downloading models')

    # Subparser for 'transcribe' command

    # add check for inputs
    parser_transcribe = subparsers.add_parser('transcribe', help='Start transcription process for an audio file')
    parser_transcribe.add_argument('file', help='Path to the audio file')
    parser_transcribe.add_argument("--model", default="large-v2", help="Model to use for transcription")
    parser_transcribe.add_argument("--language", default="auto-detect", help="Language of the audio")
    parser_transcribe.add_argument("--speaker_detection", default=False, action="store_true", help="Enable speaker detection")
    parser_transcribe.add_argument("--num_speakers", default="auto-detect", help="Number of speakers")
    parser_transcribe.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Device to use (CPU/GPU)")
    parser_transcribe.add_argument("--compute_type", default="int8", choices=["float16", "int8"], help="Compute type (float16/int8)")

    args = parser.parse_args()

    file, model, language, speaker_detection, num_speakers, device, compute_type = args.file, args.model, args.language, args.speaker_detection, args.num_speakers, args.device, args.compute_type

    if args.command == "load":
        print("Downloading all models:")
        download_all_resources()
        print("Finished")

    elif args.command == "transcribe":
        print("Running aTrain_core")
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        file_id = create_file_id(file, timestamp)
     
        try:
            check_inputs_transcribe(file, model, language)
            transcribe(file, file_id, model, language, speaker_detection, num_speakers, device, compute_type, timestamp)
        except Exception as error:
            delete_transcription(file_id)
            traceback_str = traceback.format_exc()
            error = str(error)
            print(f"The following error has occured: {error}")
            print(f"Traceback: {traceback_str}")

if __name__ == "__main__":
    cli()


