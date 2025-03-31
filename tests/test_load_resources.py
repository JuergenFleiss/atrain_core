import unittest
from unittest.mock import patch, mock_open, MagicMock, call
from aTrain_core.load_resources import (
    download_all_models,
    load_model_config_file,
    get_model,
    remove_model,
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

    @patch("aTrain_core.load_resources.snapshot_download")
    @patch("aTrain_core.load_resources.get_total_model_download_steps", return_value=10)
    @patch("aTrain_core.load_resources.ProgressTracker")
    @patch("os.path.exists", return_value=False)
    @patch("os.path.join", return_value="mock/model/path")
    @patch(
        "aTrain_core.load_resources.load_model_config_file",
        return_value={"test_model": {"repo_id": "repo", "revision": "v1"}},
    )
    def test_get_model_download(
        self,
        mock_load_model_config_file,
        mock_path_join,
        mock_path_exists,
        mock_progress_tracker,
        mock_get_total_steps,
        mock_snapshot_download,
    ):
        """Test downloading a model when it does not exist."""
        mock_gui = MagicMock()
        mock_tracker = mock_progress_tracker.return_value
        mock_tracker.progress_callback.return_value = {"current": 5, "total": 10}

        model_path = get_model("test_model", GUI=mock_gui)

        mock_snapshot_download.assert_called_once_with(
            repo_id="repo",
            revision="v1",
            local_dir="mock/model/path",
            local_dir_use_symlinks=False,
            progress_callback=unittest.mock.ANY,  # Ignore the callback comparison
        )

        progress_cb = mock_snapshot_download.call_args[1]["progress_callback"]
        progress_cb(5)  # Simulate the callback being called with a chunk

        mock_gui.progress_info.assert_called_with(current=5, total=10)

        self.assertEqual(model_path, "mock/model/path")

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


if __name__ == "__main__":
    unittest.main()
