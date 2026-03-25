"""Unit tests for configuration loading."""

from offroad_autonomy.utils.config import load_config


def test_load_config_reads_perception_prompts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "perception:",
                '  model_weights: "dummy.pt"',
                "  prompts:",
                '    - "trail"',
                '    - "road"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model_weights == "dummy.pt"
    assert config.perception_prompts == ["trail", "road"]
