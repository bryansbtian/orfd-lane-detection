"""Presentation Claims Must Reflect The Supplied Measurement Batches."""

from offroad_autonomy.visualization.route_presentation import RoutePresentation


def test_results_show_failed_attempts_and_measured_speed(monkeypatch):
    renderer = RoutePresentation.__new__(RoutePresentation)
    renderer.route = {"mean_absolute_offset_target_m": 0.5, "std_signed_offset_target_m": 0.5}
    renderer.baseline = {
        "pooled_mean_absolute_lateral_offset_m": 0.9,
        "pooled_std_signed_lateral_offset_m": 0.8,
        "maximum_absolute_lateral_offset_m": 1.5,
    }
    renderer.improved = {
        "pooled_mean_absolute_lateral_offset_m": 0.3,
        "pooled_std_signed_lateral_offset_m": 0.4,
        "maximum_absolute_lateral_offset_m": 1.2,
        "measured_attempts": 3,
        "completed_runs": 1,
        "manual_interventions": 2,
        "road_departures": 4,
        "maximum_speed_mph": 8.125,
    }
    labels = []
    monkeypatch.setattr(renderer, "words", lambda canvas, text, *args: labels.append(text))
    renderer.results()
    text = "\n".join(labels)
    assert "1/3 Routes Completed" in text
    assert "2 Recorded Interventions" in text
    assert "4 Authored-Corridor Departures" in text
    assert "8.125 mph" in text
    assert "Targets Met: 2/2" in text
    assert "5/5" not in text
    assert "6.692" not in text
    assert "Worsened" not in text
