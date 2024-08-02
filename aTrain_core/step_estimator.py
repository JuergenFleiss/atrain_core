import math

def calculate_steps(speaker_detection, nr_segments, audio_duration):
    """Calculates the total number of steps for the transcription process."""

    total_steps = 0
    if not speaker_detection:
        total_steps = nr_segments
        print(f"Total steps without diarization: {total_steps}")
        return total_steps

    elif speaker_detection:
        total_steps += nr_segments
        # Predictions
        segmentation_prediction = predict_segmentation_steps(audio_duration)
        embedding_prediction = predict_embedding_steps(audio_duration)
        segmentation_prediction = segmentation_prediction/32
        # account for one extra process when not divisible by 32
        if isinstance(segmentation_prediction/32, int):
            segmentation_prediction = segmentation_prediction
        else:
            segmentation_prediction = math.ceil(segmentation_prediction+1)

        total_steps += segmentation_prediction + embedding_prediction + 2 # speaker_counting & discrete_diarization are one step each
        return total_steps

def predict_segmentation_steps(length):
    """A function that estimates the number of computational steps during the segmentation process."""
    # These coefficients come from running a quadratic regression on data form some sample files.
    a = 1959430015971981 / 1180591620717411303424
    b = 4499524351105379 / 2251799813685248
    c = -1060416894995783 / 140737488355328
    return a * length**2 + b * length + c

def predict_embedding_steps(length):
    """A function that estimates the number of computational steps during the embedding process."""
    # These coefficients come from running a quadratic regression on data form some sample files.
    a = -6179382659256857/9444732965739290427392
    b = 6779072611703841/36028797018963968
    c = -3802017452395983/18014398509481984
    return a * length**2 + b * length + c

def get_total_model_download_steps(model):
    """A function that finds the total download chunks (steps) for a given model. 
    The metadata has been pre-calculated by downloading the models"""

    model_metadata_dict = [{'model_name': 'tiny',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 14,
    'model_bin_size': 75538270},
    {'model_name': 'base',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 20,
    'model_bin_size': 145217532},
    {'model_name': 'small',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 53,
    'model_bin_size': 483546902},
    {'model_name': 'medium',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 152,
    'model_bin_size': 1527906378},
    {'model_name': 'large-v1',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 301,
    'model_bin_size': 3086912962},
    {'model_name': 'large-v2',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 301,
    'model_bin_size': 3086912962},
    {'model_name': 'large-v3',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 301,
    'model_bin_size': 3087284237},
    {'model_name': 'faster-distil-english',
    'num_filtered_files': 7,
    'download_chunk_size': 10485760,
    'chunks_total': 151,
    'model_bin_size': 1512556667},
    {'model_name': 'diarize',
    'num_filtered_files': 5,
    'download_chunk_size': 10485760,
    'chunks_total': 14,
    'model_bin_size': 114102729}]

    total_chunks = model_metadata_dict[model]['chunks_total']
    return total_chunks
