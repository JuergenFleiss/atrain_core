import os
import json
from importlib.resources import files
import platform

def check_inputs_transcribe(file, model, language, device):
    """Check the validity of inputs for the transcription process.

    Args:
        file (str): Path to the audio file.
        model (str): Model to use for transcription.
        language (str): Language of the audio.

    Raises:
        ValueError: If any of the inputs is incorrect.
    """

    file_correct = check_file(file)
    model_correct = check_model(model, language)
    language_correct = check_language(language)
    device_correct = check_device(device)

    if not file_correct and model_correct and language_correct:
        raise ValueError("Incorrect input. Please check the file, model and language inputs.")

def check_file(file):
    """Check if the provided file is in a correct format for transcription.

    Args:
        file (str): Path to the audio file.

    Returns:
        bool: True if the file format is correct, False otherwise.
    """
    #if isinstance(file, str):
    filename = file
    # else:
    #     # If 'file' is a FileStorage object (from Flask file upload),
    #     filename = file.filename
    file_extension = os.path.splitext(filename)[1]
    file_extension_lower = str(file_extension).lower()
    # try pyav and return error
    correct_file_formats = ['.3dostr', '.4xm', '.aa', '.aac', '.ac3', '.acm', '.act', '.adf', '.adp', '.ads', '.adx', '.aea', '.afc', '.aiff', '.aix', '.alaw', '.alias_pix', '.alp', '.amr', '.amrnb', '.amrwb', '.anm', '.apc', '.ape', '.apm', '.apng', '.aptx', '.aptx_hd', '.aqtitle', '.argo_asf', '.asf', '.asf_o', '.ass', '.ast', '.au', '.av1', '.avi', '.avr', '.avs', '.avs2', '.bethsoftvid', '.bfi', '.bfstm', '.bin', '.bink', '.bit', '.bmp_pipe', '.bmv', '.boa', '.brender_pix', '.brstm', '.c93', '.caf', '.cavsvideo', '.cdg', '.cdxl', '.cine', '.codec2', '.codec2raw', '.concat', '.data', '.daud', '.dcstr', '.dds_pipe', '.derf', '.dfa', '.dhav', '.dirac', '.dnxhd', '.dpx_pipe', '.dsf', '.dsicin', '.dss', '.dts', '.dtshd', '.dv', '.dvbsub', '.dvbtxt', '.dxa', '.ea', '.ea_cdata', '.eac3', '.epaf', '.exr_pipe', '.f32be', '.f32le', '.f64be', '.f64le', '.fbdev', '.ffmetadata', '.film_cpk', '.filmstrip', '.fits', '.flac', '.flic', '.flv', '.frm', '.fsb', '.fwse', '.g722', '.g723_1', '.g726', '.g726le', '.g729', '.gdv', '.genh', '.gif', '.gif_pipe', '.gsm', '.gxf', '.h261', '.h263', '.h264', '.hca', '.hcom', '.hevc', '.hls', '.hnm', '.ico', '.idcin', '.idf', '.iff', '.ifv', '.ilbc', '.image2', '.image2pipe', '.ingenient', '.ipmovie', '.ircam', '.iss', '.iv8', '.ivf', '.ivr', '.j2k_pipe', '.jacosub', '.jpeg_pipe', '.jpegls_pipe', '.jv', '.kux', '.kvag', '.lavfi', '.live_flv', '.lmlm4', '.loas', '.lrc', '.lvf', '.lxf', '.m4v', '.matroska', '.webm', '.mgsts', '.microdvd', '.mjpeg', '.mjpeg_2000', '.mlp', '.mlv', '.mm', '.mmf', '.mov', '.mp4', '.m4a', '.3gp', '.3g2', '.mj2', '.mkv', '.mp3', '.mpc', '.mpc8', '.mpeg', '.mpegts', '.mpegtsraw', '.mpegvideo', '.mpjpeg', '.mpl2', '.mpsub', '.msf', '.msnwctcp', '.mtaf', '.mtv', '.mulaw', '.musx', '.mv', '.mvi', '.mxf', '.mxg', '.nc', '.nistsphere', '.nsp', '.nsv', '.nut', '.nuv', '.ogg', '.oma', '.oss', '.paf', '.pam_pipe', '.pbm_pipe', '.pcx_pipe', '.pgm_pipe', '.pgmyuv_pipe', '.pictor_pipe', '.pjs', '.pmp', '.png_pipe', '.pp_bnk', '.ppm_pipe', '.psd_pipe', '.psxstr', '.pva', '.pvf', '.qcp', '.qdraw_pipe', '.r3d', '.rawvideo', '.realtext', '.redspark', '.rl2', '.rm', '.roq', '.rpl', '.rsd', '.rso', '.rtp', '.rtsp', '.s16be', '.s16le', '.s24be', '.s24le', '.s32be', '.s32le', '.s337m', '.s8', '.sami', '.sap', '.sbc', '.sbg', '.scc', '.sdp', '.sdr2', '.sds', '.sdx', '.ser', '.sgi_pipe', '.shn', '.siff', '.sln', '.smjpeg', '.smk', '.smush', '.sol', '.sox', '.spdif', '.srt', '.stl', '.subviewer', '.subviewer1', '.sunrast_pipe', '.sup', '.svag', '.svg_pipe', '.swf', '.tak', '.tedcaptions', '.thp', '.tiertexseq', '.tiff_pipe', '.tmv', '.truehd', '.tta', '.tty', '.txd', '.ty', '.u16be', '.u16le', '.u24be', '.u24le', '.u32be', '.u32le', '.u8', '.v210', '.v210x', '.vag', '.vc1', '.vc1test', '.vidc', '.video4linux2', '.v4l2', '.vividas', '.vivo', '.vmd', '.vobsub', '.voc', '.vpk', '.vplayer', '.vqf', '.w64', '.wav', '.wc3movie', '.webm_dash_manifest', '.webp_pipe', '.webvtt', '.wsaud', '.wsd', '.wsvqa', '.wtv', '.wv', '.wve', '.xa', '.xbin', '.xmv', '.xpm_pipe', '.xvag', '.xwd_pipe', '.xwma', '.yop', '.yuv4mpegpipe']
    return file_extension_lower in correct_file_formats


def check_device(device):
    system = platform.system()
    if system in ["Windows", "Linux"]:
        if device == "GPU":
            return device
    elif system == "Darwin":
        if device == "GPU":
            raise ValueError("GPU is not supported on MacOS. Please choose 'CPU' instead.")

def check_model(model, language):
    """Check if the provided model and language are valid for transcription.

    Args:
        model (str): Model to use for transcription.
        language (str): Language of the audio.

    Returns:
        bool: True if the model and language are valid, False otherwise.

    Raises:
        ValueError: If the model or language is not available or if the language is not supported by the model.
    """
    # better to look into models.json and check if available
    models_config_path = str(files("aTrain_core.models").joinpath("models.json"))
    f = open(models_config_path, "r")
    models = json.load(f)
    available_models = []
    for key in models.keys():
        available_models.append(key)
    
    if model not in available_models:
        raise ValueError(f"Model {model} is not available. These are the available models: {available_models} (Note: model 'diarize' is for speaker detection only)")
    
    if models[model]["type"] == "regular":
        #print(f" model: {models[model]['type']} is used")
        return model in available_models
    
    elif models[model]["type"] == "distil":
        #print(f" model: {models[model]['type']} is used")
        if language != models[model]["language"]:
            raise ValueError(f"Language input wrong or unspecified. This distil model is only available in {models[model]['language']} and has to be specified.")
        else:
            print("Distil model and language chosen successfully")
            return model in available_models
    

def check_language(language):
    """Check if the provided language is supported for transcription.

    Args:
        language (str): Language of the audio.

    Returns:
        bool: True if the language is supported, False otherwise.
    """
    # based on which model and double-check with models.json and error if wrong language (e.g. distilled english in french)
    # auto-detect not for distilled models
    # implement check in json if language key is present
    correct_languages = ["auto-detect", "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"]

    return language in correct_languages