import unittest
from unittest.mock import patch
import numpy as np
from aTrain_core.transcribe import transcribe, EventSender


class TestTranscribe(unittest.TestCase):
    @patch("aTrain_core.transcribe.decode_audio")
    @patch("aTrain_core.transcribe.get_model")
    @patch("aTrain_core.transcribe._perform_whisper_transcription")
    @patch("aTrain_core.transcribe._perform_pyannote_speaker_diarization")
    @patch("aTrain_core.transcribe.create_metadata")
    @patch("aTrain_core.transcribe.create_output_files")
    @patch("aTrain_core.transcribe.write_logfile")
    @patch("aTrain_core.transcribe.add_processing_time_to_metadata")
    def test_transcribe_no_speaker_detection(
        self,
        mock_add_processing_time,
        mock_write_logfile,
        mock_create_output_files,
        mock_create_metadata,
        mock_perform_pyannote_speaker_diarization,
        mock_perform_whisper_transcription,
        mock_get_model,
        mock_decode_audio,
    ):
        audio_file = "test_audio.wav"
        file_id = "12345"
        model = "small"
        language = "en"
        speaker_detection = False
        num_speakers = 1
        device = "CPU"
        compute_type = "float32"
        timestamp = "2024-01-01 00:00:00"
        original_audio_filename = "original.wav"
        gui = EventSender()

        mock_decode_audio.return_value = np.array([0.0])  # Dummy audio array
        mock_get_model.return_value = "path/to/model"
        mock_perform_whisper_transcription.return_value = {"segments": []}

        transcribe(
            audio_file,
            file_id,
            model,
            language,
            speaker_detection,
            num_speakers,
            device,
            compute_type,
            timestamp,
            original_audio_filename,
            GUI=gui,
        )

        mock_decode_audio.assert_called_once_with(audio_file, sampling_rate=16000)
        mock_create_metadata.assert_called_once()
        mock_perform_whisper_transcription.assert_called_once()
        mock_create_output_files.assert_called_once()
        mock_perform_pyannote_speaker_diarization.assert_not_called()
        mock_add_processing_time.assert_called_once_with(file_id)

    @patch("aTrain_core.transcribe.decode_audio")
    @patch("aTrain_core.transcribe.get_model")
    @patch("aTrain_core.transcribe._perform_whisper_transcription")
    @patch("aTrain_core.transcribe._perform_pyannote_speaker_diarization")
    @patch("aTrain_core.transcribe.create_metadata")
    @patch("aTrain_core.transcribe.create_output_files")
    @patch("aTrain_core.transcribe.write_logfile")
    @patch("aTrain_core.transcribe.add_processing_time_to_metadata")
    def test_transcribe_with_speaker_detection(
        self,
        mock_add_processing_time,
        mock_write_logfile,
        mock_create_output_files,
        mock_create_metadata,
        mock_perform_pyannote_speaker_diarization,
        mock_perform_whisper_transcription,
        mock_get_model,
        mock_decode_audio,
    ):
        audio_file = "test_audio.wav"
        file_id = "12345"
        model = "whisper-model"
        language = "en"
        speaker_detection = True
        num_speakers = 2
        device = "CPU"
        compute_type = "float32"
        timestamp = "2024-01-01 00:00:00"
        original_audio_filename = "original.wav"
        gui = EventSender()

        mock_decode_audio.return_value = np.array([0.0])
        mock_get_model.return_value = "path/to/model"
        mock_perform_whisper_transcription.return_value = {"segments": []}
        mock_perform_pyannote_speaker_diarization.return_value = {"segments": []}

        transcribe(
            audio_file,
            file_id,
            model,
            language,
            speaker_detection,
            num_speakers,
            device,
            compute_type,
            timestamp,
            original_audio_filename,
            GUI=gui,
        )

        mock_decode_audio.assert_called_once_with(audio_file, sampling_rate=16000)
        mock_create_metadata.assert_called_once()
        mock_perform_whisper_transcription.assert_called_once()
        mock_perform_pyannote_speaker_diarization.assert_called_once()
        mock_create_output_files.assert_called_once()
        mock_add_processing_time.assert_called_once_with(file_id)

    @patch("aTrain_core.transcribe.write_logfile")
    def test_invalid_audio_file(self, mock_write_logfile):
        audio_file = "invalid_audio.wav"
        file_id = "12345"
        model = "small"
        language = "en"
        speaker_detection = False
        num_speakers = 1
        device = "CPU"
        compute_type = "float32"
        timestamp = "2024-01-01 00:00:00"
        original_audio_filename = "original.wav"
        gui = EventSender()

        with patch(
            "aTrain_core.transcribe.decode_audio",
            side_effect=Exception("No audio found."),
        ):
            with self.assertRaises(Exception) as context:
                transcribe(
                    audio_file,
                    file_id,
                    model,
                    language,
                    speaker_detection,
                    num_speakers,
                    device,
                    compute_type,
                    timestamp,
                    original_audio_filename,
                    GUI=gui,
                )
            self.assertEqual(
                str(context.exception), "Attention: Your file has no audio."
            )


if __name__ == "__main__":
    unittest.main()
