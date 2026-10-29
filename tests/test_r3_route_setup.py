"""Fresh Checkouts Can Recreate Evaluation Geometry Without Replacing Evidence."""

import json
import runpy
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def select_geometry():
    script = Path(__file__).resolve().parents[1] / "scripts/r3_centerline.py"
    return runpy.run_path(str(script))["select_geometry"]


def write_inputs(tmp_path, *, survey_map="test_map", road_ids=("selected",)):
    edges = [{"middle": [x, 0, 0], "left": [x, 3, 0], "right": [x, -3, 0]} for x in (0, 10)]
    survey = tmp_path / "survey.json"
    survey.write_text(
        json.dumps(
            {
                "map": survey_map,
                "recorded_utc": "Synthetic Fixture",
                "roads": {
                    str(i): {"persistentId": road_id, "edges": edges}
                    for i, road_id in enumerate(road_ids)
                },
            }
        ),
        encoding="utf-8",
    )
    route = tmp_path / "route.yaml"
    route.write_text(
        yaml.safe_dump(
            {
                "map": "test_map",
                "road_persistent_id": "selected",
                "geometry_file": "results/road.json",
            }
        ),
        encoding="utf-8",
    )
    return survey, route


def test_road_selection_produces_evaluation_snapshot_and_preserves_it(tmp_path, select_geometry):
    survey, route = write_inputs(tmp_path, road_ids=("other", "selected"))
    output = select_geometry(survey, route)
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["road"]["persistentId"] == "selected"
    assert snapshot["road_id_at_survey"] == "1"
    assert len(snapshot["road"]["edges"]) == 2
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        select_geometry(survey, route)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "kwargs", [{"survey_map": "wrong"}, {"road_ids": ()}, {"road_ids": ("selected", "selected")}]
)
def test_invalid_survey_cannot_create_geometry_snapshot(tmp_path, select_geometry, kwargs):
    survey, route = write_inputs(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        select_geometry(survey, route)
    assert not (tmp_path / "results/road.json").exists()
