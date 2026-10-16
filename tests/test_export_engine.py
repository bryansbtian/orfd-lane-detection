from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts import export_engine


@pytest.mark.parametrize("task", ["semantic", "segment"])
def test_export_preserves_fixed_semantic_classes_and_prompts_yoloe(task):
    model = Mock(task=task)
    model.export.return_value = "model.engine"
    config = SimpleNamespace(perception_prompts=["traversable road"], perception_input_size=1024)
    with (
        patch("sys.argv", ["export_engine", "--weights", "model.pt"]),
        patch.object(export_engine, "YOLO", return_value=model),
        patch.object(export_engine, "load_config", return_value=config),
    ):
        export_engine.main()
    if task == "semantic":
        model.set_classes.assert_not_called()
    else:
        model.set_classes.assert_called_once_with(config.perception_prompts)
    model.export.assert_called_once_with(format="engine", imgsz=1024, half=True, device=0)
