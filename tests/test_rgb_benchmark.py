"""The label-based scorer must not feed annotations back into inference."""

import json
from unittest.mock import Mock

import cv2
import numpy as np

from offroad_autonomy.types import PathPlan, PerceptionResult, PipelineConfig
from scripts import benchmark_rgb


def test_planning_replay_uses_camera_roi_and_keeps_annotations_out_of_inference(
    tmp_path, monkeypatch
):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "1000.png"), np.zeros((620, 960, 3), np.uint8))
    cv2.imwrite(str(labels / "1000_fillcolor.png"), np.full((620, 960), 255, np.uint8))
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"mock weights")
    config = PipelineConfig(model_weights=str(weights))
    output = tmp_path / "report.json"
    segmenter = Mock()

    def predict(packet, valid_roi):
        assert not packet.preprocessed.any()
        assert valid_roi.shape == (465, 720)
        assert not valid_roi.all()
        return PerceptionResult(mask=valid_roi.copy(), confidences=[0.9], valid_roi=valid_roi)

    segmenter.predict.side_effect = predict
    planner = Mock()
    planner.plan.return_value = PathPlan(centerline=np.empty((0, 2)), fallback_active=True)
    monkeypatch.setattr(benchmark_rgb, "load_config", lambda path: config)
    monkeypatch.setattr(benchmark_rgb, "RoadSegmenter", lambda cfg: segmenter)
    monkeypatch.setattr(benchmark_rgb, "CenterlinePlanner", lambda *args, **kwargs: planner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_rgb",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--out",
            str(output),
            "--planning",
            "--warmup",
            "0",
        ],
    )
    benchmark_rgb.main()
    segmenter.predict.assert_called_once()
    report = json.loads(output.read_text())
    assert report["planning_enabled"]
    assert not report["annotations_used_for_inference"]
    assert report["raw"]["frames"] == 1
