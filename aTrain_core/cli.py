from .transcribe import transcribe
from .check_inputs import check_inputs_transcribe
from .outputs import create_file_id, delete_transcription
from .transcribe import transcribe
from .globals import TIMESTAMP_FORMAT
from .load_resources import download_all_resources
import traceback
from datetime import datetime
import argparse

def link(uri, label=None):
    if label is None: 
        label = uri
    parameters = ''

    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST 
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)



def cli():
    """Command-line interface for running audio transcription with Whisper using aTrain_core.

    This CLI tool allows users to transcribe audio files using the Whisper model. It provides two main commands:

    1. 'load': Initializes aTrain_core by downloading required models.
    2. 'transcribe': Starts the transcription process for an audio file.

    Command usage:
    1. To initialize aTrain_core and download all required models:
        aTrain_core load

    2. To transcribe an audio file:
        aTrain_core transcribe <path_to_audio_file> [--model MODEL] [--language LANGUAGE]
                                                     [--speaker_detection] [--num_speakers NUM_SPEAKERS]
                                                     [--device DEVICE] [--compute_type COMPUTE_TYPE]

        Arguments:
          - <path_to_audio_file>: Path to the audio file to transcribe.
          - --model MODEL: Model to use for transcription (default is 'large-v2').
          - --language LANGUAGE: Language of the audio (default is 'auto-detect').
          - --speaker_detection: Enable speaker detection (optional).
          - --num_speakers NUM_SPEAKERS: Number of speakers (default is 'auto-detect').
          - --device DEVICE: Device to use (options are 'CPU' or 'GPU', default is 'CPU').
          - --compute_type COMPUTE_TYPE: Compute type (options are 'float16' or 'int8', default is 'int8').

    Note: If an error occurs during transcription, the tool automatically deletes the partially created transcription.

    """
    
    parser = argparse.ArgumentParser(prog='aTrain_core', description='A CLI tool for audio transcription with Whisper')
    subparsers = parser.add_subparsers(dest='command', help='Command for aTrain_core to perform.')

    # Subparser for 'init' command
    parser_load = subparsers.add_parser('load', help='Initialize aTrain_core by downloading models')

    # Subparser for 'transcribe' command

    # add check for inputs
    parser_transcribe = subparsers.add_parser('transcribe', help='Start transcription process for an audio file')
    parser_transcribe.add_argument('file', help='Path to the audio file')
    parser_transcribe.add_argument("--model", default="large-v3", help="Model to use for transcription")
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
            print(f"Thank you for using aTrain \nIf you use aTrain in a scientific publication, please consider citing our paper:\n'Take the aTrain. Introducing an interface for the Accessible Transcription of Interviews'\navailable under: {link('https://www.sciencedirect.com/science/article/pii/S2214635024000066')}")
        except Exception as error:
            delete_transcription(file_id)
            traceback_str = traceback.format_exc()
            error = str(error)
            print(f"The following error has occured: {error}")
            print(f"Traceback: {traceback_str}")

if __name__ == "__main__":
    cli()


