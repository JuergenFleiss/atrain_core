import unittest
import os
from unittest.mock import patch, mock_open, MagicMock, call
from aTrain_core.load_resources import (
    download_all_models,
    load_model_config_file,
    get_model,
    remove_model,
    assert_model_hash,
)
from aTrain_core.globals import MODELS_DIR


class TestLoadResourcesFunctions(unittest.TestCase):
    @patch("aTrain_core.load_resources.load_model_config_file")
    @patch("aTrain_core.load_resources.get_model")
    def test_download_all_models(self, mock_get_model, mock_load_model_config_file):
        """Test download_all_models calls get_model for each model in config."""
        mock_load_model_config_file.return_value = {
            "base": {
                "repo_id": "aTrain-core/faster-whisper-base",
                "revision": "60807e2d2c25d190ceda6bb43b6f8161f98e60da",
                "type": "regular",
                "num_filtered_files": 7,
                "download_chunk_size": 10485760,
                "chunks_total": 20,
                "model_bin_size": 145217532,
                "model_bin_size_human": "145.22 MB",
            },
            "small": {
                "repo_id": "aTrain-core/faster-whisper-small",
                "revision": "89b1eeb2b813706ff13f183df2ec32a2d62edc35",
                "type": "regular",
                "num_filtered_files": 7,
                "download_chunk_size": 10485760,
                "chunks_total": 53,
                "model_bin_size": 483546902,
                "model_bin_size_human": "483.55 MB",
            },
        }
        download_all_models()
        mock_get_model.assert_has_calls([call("base"), call("small")])
        self.assertEqual(mock_get_model.call_count, 2)

    @patch("builtins.open", new_callable=mock_open, read_data='{"small": {}}')
    @patch("aTrain_core.load_resources.files")
    def test_load_model_config_file(self, mock_files, mock_open_file):
        """Test if model config file loads properly."""
        mock_files.return_value = MagicMock(
            joinpath=MagicMock(return_value=f"{MODELS_DIR}/models.json")
        )
        models_config = load_model_config_file()
        mock_open_file.assert_called_once_with(f"{MODELS_DIR}/models.json", "r")
        self.assertEqual(models_config, {"small": {}})

    @patch("os.path.exists", return_value=True)
    @patch("shutil.rmtree")
    @patch("os.path.join", return_value="mock/model/path")
    def test_remove_model(self, mock_path_join, mock_rmtree, mock_path_exists):
        """Test removing a model."""
        remove_model("test_model")
        mock_rmtree.assert_called_once_with("mock/model/path")

    @patch("os.path.exists", return_value=False)
    @patch("shutil.rmtree")
    @patch("os.path.join", return_value="mock/model/path")
    def test_remove_model_does_not_exist(
        self, mock_path_join, mock_rmtree, mock_path_exists
    ):
        """Test removing a model that doesn't exist."""
        remove_model("test_model")
        mock_rmtree.assert_not_called()

    def test_assert_model_hash_valid(self):
        # Define valid inputs
        model_info = {"model_hash": "correct_hash"}
        dir_hash = "correct_hash"

        # No exception should be raised with valid hash
        try:
            assert_model_hash(
                dir_hash, "test_model", model_info, False, "model_path", "required_path"
            )
        except AssertionError:
            self.fail("assert_model_hash raised AssertionError unexpectedly!")

    @patch("aTrain_core.load_resources.remove_model")
    def test_assert_model_hash_invalid(self, mock_remove_model):
        # Define invalid inputs
        model_info = {"model_hash": "correct_hash"}
        dir_hash = "wrong_hash"

        # Assert that an exception is raised and model is removed
        with self.assertRaises(AssertionError):
            assert_model_hash(
                dir_hash, "test_model", model_info, False, "model_path", "required_path"
            )

        mock_remove_model.assert_called_once_with("test_model", "model_path")

    @patch("os.path.exists")
    @patch("checksumdir.dirhash")
    @patch("shutil.rmtree")  # Mock shutil.rmtree to avoid the FileNotFoundError
    def test_get_model_existing(self, mock_rmtree, mock_dirhash, mock_path_exists):
        # Mock os.path.exists: return False for the .cache path and True for everything else
        mock_path_exists.side_effect = (
            lambda path: path != "model_path/test_model/.cache"
        )
        mock_dirhash.return_value = "fake_hash"

        # Mock the model info and configs
        with patch(
            "aTrain_core.load_resources.load_model_config_file"
        ) as mock_load_config, patch(
            "aTrain_core.load_resources.MODELS_DIR", "model_path"
        ), patch("aTrain_core.load_resources.REQUIRED_MODELS", ["required_model"]):
            mock_load_config.return_value = {
                "test_model": {
                    "repo_id": "fake_repo",
                    "revision": "v1",
                    "model_hash": "fake_hash",
                }
            }
            GUI = MagicMock()

            # Call the function
            model_path = get_model("test_model", GUI)

            # Assertions
            GUI.progress_info.assert_not_called()  # No download should occur if the model exists
            mock_dirhash.assert_called_once()  # Ensure hash was checked
            mock_rmtree.assert_called_once()

    @patch("os.path.exists")
    @patch("checksumdir.dirhash")
    @patch("shutil.rmtree")
    @patch("aTrain_core.load_resources.snapshot_download")  # Mock the download process
    def test_get_model_download(
        self, mock_snapshot_download, mock_rmtree, mock_dirhash, mock_path_exists
    ):
        # Simulate the model directory and .cache not existing to trigger the download
        mock_path_exists.side_effect = (
            lambda path: path == "model_path/test_model/.cache"
        )
        mock_dirhash.return_value = "fake_hash"

        # Mock the model info and configs
        with patch(
            "aTrain_core.load_resources.load_model_config_file"
        ) as mock_load_config, patch(
            "aTrain_core.load_resources.MODELS_DIR", "model_path"
        ), patch("aTrain_core.load_resources.REQUIRED_MODELS", ["required_model"]):
            mock_load_config.return_value = {
                "test_model": {
                    "repo_id": "fake_repo",
                    "revision": "v1",
                    "model_hash": "fake_hash",
                }
            }

            # Mock the progress tracker
            GUI = MagicMock()

            # Simulate total chunks for the download process
            with patch(
                "aTrain_core.load_resources.get_total_model_download_steps"
            ) as mock_total_steps:
                mock_total_steps.return_value = (
                    100  # Assume 100 chunks for the download
                )

                # Call the function
                model_path = get_model("test_model", GUI)

                # Assertions
                mock_snapshot_download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
