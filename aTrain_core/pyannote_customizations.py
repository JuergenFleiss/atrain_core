import os
from typing import Any, Mapping, Optional, Text

import yaml
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.utils.helper import get_class_by_name

from .GUI_integration import EventSender


class CustomPipeline(Pipeline):
    @classmethod
    def from_pretrained(cls, model_path, required_models_dir) -> Pipeline:
        """Constructs a custom pipeline from pre-trained models."""
        config_yml = os.path.join(required_models_dir, "diarize", "config.yaml")
        with open(config_yml, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        params = config["pipeline"].get("params", {})
        path_segmentation_model = os.path.join(model_path, "segmentation_pyannote.bin")
        path_embedding_model = os.path.join(model_path, "embedding_pyannote.bin")
        params["segmentation"] = path_segmentation_model.replace("\\", "/")
        params["embedding"] = path_embedding_model.replace("\\", "/")
        pipeline: Pipeline = Klass(**params)
        pipeline.instantiate(config["params"])
        return pipeline


class CustomProgressHook(ProgressHook):
    """A custom progress hook that updates the GUI and prints progress information during processing."""

    def __init__(self, GUI: EventSender, completed_steps, total_steps):
        super().__init__()
        self.GUI = GUI
        self.completed_steps = completed_steps
        self.total_steps = total_steps

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        super().__call__(step_name, step_artifact, file, total, completed)

        # self.GUI.task_info(f"{self.step_name}")    # names of sub-steps within speaker detection
        self.GUI.task_info("Detect Speakers")

        if (
            self.step_name == "speaker_counting"
            or self.step_name == "discrete_diarization"
        ):
            self.completed_steps += 1
            self.GUI.progress_info(self.completed_steps, self.total_steps)

        if total is not None and completed is not None:
            if completed != 0:
                self.completed_steps += 1
                self.GUI.progress_info(self.completed_steps, self.total_steps)
