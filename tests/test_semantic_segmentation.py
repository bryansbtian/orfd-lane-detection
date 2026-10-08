from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.perception.semantic_predictor import RoadSemanticPredictor
from offroad_autonomy.types import FramePacket, PipelineConfig


def _segmenter(names=None, threshold=0.15):
    model = Mock(task="semantic", names=names or {0: "background", 1: "traversable road"})
    with patch("offroad_autonomy.perception.road_segmenter.YOLO", return_value=model):
        segmenter = RoadSegmenter(PipelineConfig(confidence_threshold=threshold))
    model.set_classes.assert_not_called()
    return segmenter, model


def test_semantic_road_selection_threshold_and_ego_exclusion():
    seg, model = _segmenter(threshold=0.7)
    model.predict.return_value = [
        SimpleNamespace(
            semantic_mask=SimpleNamespace(data=torch.tensor([[0, 1, 1], [1, 1, 0]])),
            semantic_confidence=torch.tensor([[0.99, 0.8, 0.6], [0.95, 0.9, 0.99]]),
        )
    ]
    roi = np.array([[True, True, True], [False, True, True]])
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    packet = FramePacket(raw=image, preprocessed=image, timestamp=0, height=2, width=3)
    result = seg.predict(packet, roi)
    np.testing.assert_array_equal(result.mask, [[False, True, False], [False, True, False]])
    assert result.confidences == pytest.approx([0.85])
    assert result.num_detections == 1
    assert result.road_fraction == pytest.approx(2 / 5)
    assert model.predict.call_args.kwargs["predictor"] is RoadSemanticPredictor


def test_unknown_semantic_class_fails_instead_of_treating_everything_as_road():
    with pytest.raises(ValueError, match="No semantic road class"):
        _segmenter({0: "background", 1: "vehicle"})


def test_single_logit_model_selects_foreground_mask_id():
    seg, _ = _segmenter({0: "traversable road"})
    assert seg._road_ids == [1]


def test_empty_semantic_result_has_no_confidence():
    seg, _ = _segmenter()
    result = seg._semantic_result(None, 2, 3, None, 5.0)
    assert not result.mask.any()
    assert result.confidences == []
    assert result.road_fraction == 0


@pytest.mark.parametrize("binary", [False, True])
def test_semantic_predictor_preserves_probability_and_removes_letterbox(binary):
    predictor = RoadSemanticPredictor.__new__(RoadSemanticPredictor)
    predictor.model = SimpleNamespace(names={0: "road"} if binary else {0: "bg", 1: "road"})
    predictor.batch = (["frame.png"],)
    logits = torch.full((1, 1 if binary else 2, 8, 8), -8.0)
    if not binary:
        logits[:, 0] = 0
    logits[:, -1, 2:6] = 2.0  # content between two rows of padding on either side
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    result = predictor.postprocess(logits, torch.zeros(1, 3, 8, 8), [image])[0]
    assert tuple(result.semantic_mask.data.shape) == (4, 8)
    assert (result.semantic_mask.data == 1).all()
    assert torch.allclose(result.semantic_confidence, torch.full((4, 8), 0.880797), atol=1e-5)


def test_instance_model_retains_prompting_and_mask_path():
    model = Mock(task="segment")
    model.predict.return_value = [
        SimpleNamespace(
            masks=SimpleNamespace(data=torch.ones(1, 2, 3)),
            boxes=SimpleNamespace(conf=torch.tensor([0.75])),
        )
    ]
    cfg = PipelineConfig()
    with patch("offroad_autonomy.perception.road_segmenter.YOLO", return_value=model):
        seg = RoadSegmenter(cfg)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    result = seg.predict(FramePacket(raw=image, preprocessed=image, timestamp=0, height=2, width=3))
    assert result.mask.all()
    assert result.confidences == [0.75]
    model.set_classes.assert_called_once_with(cfg.perception_prompts)
