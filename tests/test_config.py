import pytest

from house_price.config import load_config


def test_project_config_loads():
    config = load_config()

    assert config["random_state"] == 42
    assert config["model"]["n_estimators"] == 200
    assert config["quality_gate"]["min_r2"] == 0.75


def test_missing_config_sections_are_rejected(tmp_path):
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text("random_state: 42\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Missing configuration sections",
    ):
        load_config(bad_config)