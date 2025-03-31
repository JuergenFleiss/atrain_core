import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil
from collections import namedtuple
import json
from aTrain_core.outputs import (
    create_directory,
    create_file_id,
    create_output_files,
    create_json_file,
    create_txt_file,
    create_srt_file,
    transform_speakers_results,
    named_tuple_to_dict,
    create_metadata,
    write_logfile,
    add_processing_time_to_metadata,
    delete_transcription,
)

from aTrain_core.globals import TRANSCRIPT_DIR


class TestOutputs(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for tests."""
        self.test_dir = "test_transcript_dir"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("os.makedirs")
    def test_create_directory(self, mock_makedirs):
        file_id = "mock_file_id"
        create_directory(file_id)
        mock_makedirs.assert_called()

    def test_create_file_id(self):
        file_path = "path/to/audio.wav"
        timestamp = "2024-10-14 14-18-16"
        expected_file_id = "2410141418-audio.w"  # Adjust according to expected output
        file_id = create_file_id(file_path, timestamp)
        self.assertEqual(file_id, expected_file_id)

    @patch("aTrain_core.outputs.create_json_file")
    @patch("aTrain_core.outputs.create_txt_file")
    @patch("aTrain_core.outputs.create_srt_file")
    def test_create_output_files(
        self, mock_create_srt_file, mock_create_txt_file, mock_create_json_file
    ):
        result = {"segments": []}
        speaker_detection = False
        file_id = "mock_file_id"
        create_output_files(result, speaker_detection, file_id)
        mock_create_json_file.assert_called_once_with(result, file_id)
        self.assertEqual(mock_create_txt_file.call_count, 3)
        mock_create_srt_file.assert_called_once_with(result, file_id)

    @patch("builtins.open", new_callable=mock_open)
    @patch("aTrain_core.globals.TRANSCRIPT_DIR", new="test_transcript_dir")
    def test_create_json_file(self, mock_open):
        result = {"test": "data"}
        file_id = "mock_file_id"

        # Ensure TRANSCRIPT_DIR is mocked correctly
        with patch("aTrain_core.globals.TRANSCRIPT_DIR", self.test_dir):
            create_json_file(result, file_id)

            mock_open.assert_called_once_with(
                os.path.join(self.test_dir, file_id, "transcription.json"),
                "w",
                encoding="utf-8",
            )

            mock_open().write.assert_called_once_with(
                json.dumps(result, ensure_ascii=False)
            )

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_create_txt_file(self, mock_open):
        result = {
            "segments": [
                {"speaker": "Speaker 1", "text": "Hello.", "start": 0, "end": 1}
            ]
        }
        file_id = "mock_file_id"
        create_txt_file(result, file_id, True, True, False)
        mock_open.assert_called_once()
        # Check if the file is created correctly with speaker detection and timestamps

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_create_srt_file(self, mock_open):
        result = {"segments": [{"start": 0, "end": 1, "text": "Hello."}]}
        file_id = "mock_file_id"
        create_srt_file(result, file_id)
        mock_open.assert_called_once()
        # Check if the SRT file is created correctly

    @patch("pandas.DataFrame")
    def test_transform_speakers_results(self, mock_dataframe):
        mock_segments = MagicMock()
        mock_segments.itertracks.return_value = []
        result = transform_speakers_results(mock_segments)
        mock_dataframe.assert_called_once()

    def test_named_tuple_to_dict(self):
        # Create a named tuple
        TestNamedTuple = namedtuple("TestNamedTuple", ["field1", "field2"])
        named_tuple = TestNamedTuple(field1=1, field2=2)

        # Call the conversion function
        result = named_tuple_to_dict(named_tuple)

        # Assert the expected result
        self.assertEqual(result, {"field1": 1, "field2": 2})

    def test_nested_named_tuple(self):
        # Create a nested named tuple
        InnerTuple = namedtuple("InnerTuple", ["inner_field1", "inner_field2"])
        OuterTuple = namedtuple("OuterTuple", ["field1", "field2", "inner"])
        nested_named_tuple = OuterTuple(
            field1=1, field2=2, inner=InnerTuple(inner_field1=3, inner_field2=4)
        )

        # Call the conversion function
        result = named_tuple_to_dict(nested_named_tuple)

        # Assert the expected result
        self.assertEqual(
            result,
            {"field1": 1, "field2": 2, "inner": {"inner_field1": 3, "inner_field2": 4}},
        )

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_create_metadata(self, mock_open):
        metadata = {
            "file_id": "mock_file_id",
            "filename": "audio.wav",
            "audio_duration": 120,
            "model": "model_name",
            "language": "en",
            "speaker_detection": True,
            "num_speakers": 2,
            "device": "CPU",
            "compute_type": "int8",
            "timestamp": "2024-10-14 14-18-16",
            "original_audio_filename": "path/to/audio.wav",
        }
        create_metadata(**metadata)
        mock_open.assert_called_once()

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_logfile(self, mock_open):
        message = "This is a log message."
        file_id = "mock_file_id"
        write_logfile(message, file_id)
        mock_open.assert_called_once()

    @patch("yaml.safe_load")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_add_processing_time_to_metadata(self, mock_open, mock_safe_load):
        mock_safe_load.return_value = {"timestamp": "2024-10-14 14-18-16"}
        file_id = "mock_file_id"
        create_metadata(
            file_id=file_id,
            filename="audio.wav",
            audio_duration=120,
            model="model",
            language="en",
            speaker_detection=True,
            num_speakers=2,
            device="CPU",
            compute_type="int8",
            timestamp="2024-10-14 14-18-16",
            original_audio_filename="path/to/audio.wav",
        )
        add_processing_time_to_metadata(file_id)
        mock_open.assert_called()

    @patch("shutil.rmtree")
    def test_delete_transcription(self, mock_rmtree):
        file_id = "mock_file_id"
        create_directory(file_id)  # Create directory for deletion test
        delete_transcription(file_id)
        mock_rmtree.assert_called_once()


if __name__ == "__main__":
    unittest.main()
