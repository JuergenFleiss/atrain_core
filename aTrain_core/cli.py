from .transcribe import transcribe
from .check_inputs import check_inputs_transcribe
from .outputs import create_file_id, delete_transcription
from .transcribe import transcribe
from .globals import TIMESTAMP_FORMAT
from .load_resources import download_all_models, get_model, remove_model
import traceback
from datetime import datetime
import argparse
import os
from werkzeug.utils import secure_filename

def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)



def cli():
    """Command-line interface for running audio transcription with Whisper using aTrain_core."""
    
    parser = argparse.ArgumentParser(prog='aTrain_core', description='A CLI tool for audio transcription with Whisper')
    subparsers = parser.add_subparsers(dest='command', help='Command for aTrain_core to perform.')

    # Subparser for 'load' command
    parser_load = subparsers.add_parser('load', help='Initialize aTrain_core by downloading models')
    parser_load.add_argument("--model", help="Model to download")

    # Subparser for 'remove' command
    parser_remove = subparsers.add_parser('remove', help='Remove models')
    parser_remove.add_argument("--model", help="Model to remove")

    # Subparser for 'transcribe' command
    parser_transcribe = subparsers.add_parser('transcribe', help='Start transcription process for an audio file')
    parser_transcribe.add_argument("audiofile", help="Path to the audio file")
    parser_transcribe.add_argument("--model", default="large-v3", help="Model to use for transcription")
    parser_transcribe.add_argument("--language", default="auto-detect", help="Language of the audio")
    parser_transcribe.add_argument("--speaker_detection", default=False, action="store_true", help="Enable speaker detection")
    parser_transcribe.add_argument("--num_speakers", default="auto-detect", help="Number of speakers")
    parser_transcribe.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Device to use (CPU/GPU)")
    parser_transcribe.add_argument("--compute_type", default="int8", choices=["float16", "int8"], help="Compute type (float16/int8)")

    args = parser.parse_args()

    
    if args.command == "load":
        if args.model == "all":
            print("Downloading all models:")
            download_all_models()
            print("All models downloaded")
        else:
            print(f"Downloading model {args.model}")
            get_model(args.model)
            print(f"Model {args.model} downloaded")

    elif args.command == "remove":
        remove_model(args.model)
        print(f"Model {args.model} removed") 

    elif args.command == "transcribe":
        print("Running aTrain_core")
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        # filename = secure_filename(args.audiofile)
        # print(f"FILENAME")
        # print(filename)

        dir_name = os.path.dirname(args.audiofile)
        file_base_name = os.path.basename(args.audiofile)

        # Secure the base name (remove unsafe characters)
        secure_file_base_name = secure_filename(file_base_name)

        # Join the directory path with the secure base name to get the full path
        filename = os.path.join(dir_name, secure_file_base_name)


        file_id = create_file_id(filename, timestamp)
     
        try:
            check_inputs_transcribe(filename, args.model, args.language, args.device)
            transcribe(filename, file_id, args.model, args.language, args.speaker_detection, args.num_speakers, args.device, args.compute_type, timestamp)
            print(f"Thank you for using aTrain \nIf you use aTrain in a scientific publication, please cite our paper:\n'Take the aTrain. Introducing an interface for the Accessible Transcription of Interviews'\navailable under: {link('https://www.sciencedirect.com/science/article/pii/S2214635024000066')}")
        except Exception as error:
            delete_transcription(file_id)
            traceback_str = traceback.format_exc()
            error = str(error)
            print(f"The following error has occured: {error}")
            print(f"Traceback: {traceback_str}")

if __name__ == "__main__":
    cli()


