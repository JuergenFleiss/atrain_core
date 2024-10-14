import unittest
from unittest.mock import patch, mock_open
from aTrain_core.check_inputs import (
    check_file,
    check_model,
    check_language,
    check_device,
    check_inputs_transcribe,
)


class TestCheckInputsFunctions(unittest.TestCase):
    def test_check_file_valid(self):
        """Test check_file with valid formats."""
        valid_files = ["audio.mp3", "video.mp4", "image.wav"]
        for file in valid_files:
            self.assertTrue(check_file(file))

    def test_check_file_invalid(self):
        """Test check_file with invalid formats."""
        invalid_files = ["document.txt", "spreadsheet.xlsx", "image.jpg"]
        for file in invalid_files:
            self.assertFalse(check_file(file))

    def test_check_model_valid(self):
        """Test check_model with valid models and languages."""
        with patch(
            "builtins.open",
            mock_open(
                read_data='{"large-v3-turbo": {"type": "regular"}, "distil-english": {"type": "distil", "language": "en"}}'
            ),
        ):
            self.assertTrue(check_model("large-v3-turbo", "en"))
            self.assertTrue(check_model("distil-english", "en"))

    def test_check_model_invalid(self):
        """Test check_model with invalid models."""
        with patch(
            "builtins.open",
            mock_open(
                read_data='{"small": {"type": "regular"}, "distil_english": {"type": "distil", "language": "en"}}'
            ),
        ):
            with self.assertRaises(ValueError):
                check_model("modelC", "en")
            with self.assertRaises(ValueError):
                check_model("distil_english", "fr")  # Wrong language for distil model

    def test_check_language_valid(self):
        """Test check_language with valid languages."""
        valid_languages = ["en", "zh", "de", "auto-detect"]
        for lang in valid_languages:
            self.assertTrue(check_language(lang))

    def test_check_language_invalid(self):
        """Test check_language with invalid languages."""
        invalid_languages = ["xx", "abc", "invalid"]
        for lang in invalid_languages:
            self.assertFalse(check_language(lang))

    def test_check_device_gpu_available(self):
        """Test check_device when GPU is available."""
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(check_device("GPU"), "GPU")

    def test_check_device_gpu_unavailable(self):
        """Test check_device when GPU is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(ValueError):
                check_device("GPU")

    def test_check_inputs_transcribe_all_valid(self):
        """Test check_inputs_transcribe with valid inputs."""
        with patch(
            "builtins.open", mock_open(read_data='{"large-v3": {"type": "regular"}}')
        ):
            with patch("torch.cuda.is_available", return_value=True):
                try:
                    check_inputs_transcribe("audio.mp3", "large-v3", "en", "GPU")
                except ValueError:
                    self.fail(
                        "check_inputs_transcribe() raised ValueError unexpectedly"
                    )

    def test_check_inputs_transcribe_invalid_file(self):
        """Test check_inputs_transcribe with an invalid file."""
        with patch(
            "builtins.open", mock_open(read_data='{"large-v3": {"type": "regular"}}')
        ):
            with patch("torch.cuda.is_available", return_value=True):
                with self.assertRaises(ValueError):
                    check_inputs_transcribe("invalid_file.txt", "large-v3", "en", "GPU")


if __name__ == "__main__":
    unittest.main()
