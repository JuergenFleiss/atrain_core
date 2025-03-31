import unittest
from unittest.mock import patch
from aTrain_core.step_estimator import (
    predict_segmentation_steps,
    predict_embedding_steps,
    get_total_model_download_steps,
)


class TestModelFunctions(unittest.TestCase):
    def test_predict_segmentation_steps(self):
        """Test segmentation steps prediction."""
        length = 10
        expected = (
            (1959430015971981 / 1180591620717411303424) * length**2
            + (4499524351105379 / 2251799813685248) * length
            + (-1060416894995783 / 140737488355328)
        )
        result = predict_segmentation_steps(length)
        self.assertAlmostEqual(result, expected, places=7)

    def test_predict_embedding_steps(self):
        """Test embedding steps prediction."""
        length = 10
        expected = (
            (-6179382659256857 / 9444732965739290427392) * length**2
            + (6779072611703841 / 36028797018963968) * length
            + (-3802017452395983 / 18014398509481984)
        )
        result = predict_embedding_steps(length)
        self.assertAlmostEqual(result, expected, places=7)

    @patch(
        "aTrain_core.step_estimator.load_model_config_file",
        return_value={"small": {"chunks_total": 10}},
    )
    def test_get_total_model_download_steps(self, mock_load):
        """Test fetching the total download steps for a given model."""

        model_name = "small"
        total_steps = get_total_model_download_steps(model_name)
        self.assertEqual(total_steps, 10)

        total_steps_non_existent = get_total_model_download_steps("model_non_existent")
        self.assertIsNone(total_steps_non_existent)


if __name__ == "__main__":
    unittest.main()
