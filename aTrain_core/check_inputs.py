import os


def check_inputs_transcribe(file, model, language):
    file_correct = check_file(file)
    model_correct = check_model(model)
    language_correct = check_language(language)

    if not file_correct and model_correct and language_correct:
        raise ValueError("Incorrect input. Please check the file, model and language inputs.")

def check_file(file):
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

def check_model(model):
    # better to look into models.json and check if available
    correct_models = ["tiny", "base", "small", "medium" , "large-v1" , "large-v2", "distilled_en"]
    return model in correct_models

def check_language(language):
    # based on which model and double-check with models.json and error if wrong language (e.g. distilled english in french)
    # auto-detect not for distilled models
    # implement check in json if language key is present
    correct_languages = ['auto-detect','af', 'ar', 'hy', 'az', 'be', 'bs', 'bg', 'ca', 'zh', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'gl', 'de', 'el', 'he', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'kn', 'kk', 'ko', 'lv', 'lt', 'mk', 'ms', 'mr', 'mi', 'ne', 'no', 'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sw', 'sv', 'tl', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'cy']
    return language in correct_languages