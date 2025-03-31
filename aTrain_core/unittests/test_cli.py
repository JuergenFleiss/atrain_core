import unittest
from unittest.mock import patch


from aTrain_core.cli import cli


class TestCli(unittest.TestCase):
    @patch("aTrain_core.cli.download_all_models")
    @patch("aTrain_core.cli.get_model")
    @patch("sys.argv", ["aTrain_core", "load", "--model", "all"])
    def test_load_all_models(self, mock_get_model, mock_download_all_models):
        """Test 'load' subcommand with 'all' models."""
        cli()
        mock_download_all_models.assert_called_once()
        mock_get_model.assert_not_called()

    @patch("aTrain_core.cli.get_model")
    @patch("sys.argv", ["aTrain_core", "load", "--model", "large-v3"])
    def test_load_specific_model(self, mock_get_model):
        """Test 'load' subcommand with a specific model."""
        cli()
        mock_get_model.assert_called_once_with("large-v3")

    @patch("aTrain_core.cli.remove_model")
    @patch("sys.argv", ["aTrain_core", "remove", "--model", "large-v3"])
    def test_remove_model(self, mock_remove_model):
        """Test 'remove' subcommand."""
        cli()
        mock_remove_model.assert_called_once_with("large-v3")

    @patch("aTrain_core.cli.create_file_id")
    @patch("aTrain_core.cli.datetime")
    @patch("aTrain_core.cli.secure_filename")
    @patch("aTrain_core.cli.transcribe")
    @patch("aTrain_core.cli.check_inputs_transcribe")
    @patch("aTrain_core.cli.write_logfile")
    @patch("aTrain_core.cli.create_directory")
    def test_transcribe(
        self,
        mock_create_directory,
        mock_write_logfile,
        mock_check_inputs_transcribe,
        mock_transcribe,
        mock_secure_filename,
        mock_datetime,
        mock_create_file_id,
    ):
        mock_secure_filename.return_value = "secure_audiofile.wav"

        mock_timestamp = "2024-10-14 14-18-16"
        mock_datetime.now.return_value.strftime.return_value = mock_timestamp

        mock_create_file_id.return_value = "mock_file_id"

        test_args = ["cli.py", "transcribe", "mock/dir/mock_audiofile.wav"]
        with patch("sys.argv", test_args):
            cli()

        mock_create_file_id.assert_called_once_with(
            "mock/dir/secure_audiofile.wav", mock_timestamp
        )

        mock_create_directory.assert_called_once_with("mock_file_id")

        mock_check_inputs_transcribe.assert_called_once_with(
            "mock/dir/secure_audiofile.wav", "large-v3", "auto-detect", "CPU"
        )

        mock_transcribe.assert_called_once_with(
            "mock/dir/secure_audiofile.wav",
            "mock_file_id",
            "large-v3",
            "auto-detect",
            False,
            "auto-detect",
            "CPU",
            "int8",
            mock_timestamp,
            "mock/dir/mock_audiofile.wav",
        )

        mock_write_logfile.assert_called_once_with(
            "File ID created: mock_file_id", "mock_file_id"
        )


if __name__ == "__main__":
    unittest.main()
