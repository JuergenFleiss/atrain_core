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