from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load and validate project configuration."""

    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required_sections = {
        "random_state",
        "validation",
        "model",
        "quality_gate",
    }

    missing = required_sections - config.keys()

    if missing:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing)}"
        )

    if not isinstance(config["validation"], dict):
        raise ValueError("'validation' must be a mapping.")

    if not isinstance(config["model"], dict):
        raise ValueError("'model' must be a mapping.")

    if not isinstance(config["quality_gate"], dict):
        raise ValueError("'quality_gate' must be a mapping.")

    test_size = config["validation"].get("test_size")

    if (
        not isinstance(test_size, (int, float))
        or isinstance(test_size, bool)
        or not 0 < test_size < 1
    ):
        raise ValueError("'validation.test_size' must be between 0 and 1.")

    n_estimators = config["model"].get("n_estimators")

    if (
        not isinstance(n_estimators, int)
        or isinstance(n_estimators, bool)
        or n_estimators < 1
    ):
        raise ValueError("'model.n_estimators' must be a positive integer.")

    gate = config["quality_gate"]

    for key in ("max_mae", "max_rmse", "min_r2"):
        if key not in gate or not isinstance(gate[key], (int, float)):
            raise ValueError(
                f"'quality_gate.{key}' must be a numeric value."
            )

    if gate["max_mae"] <= 0 or gate["max_rmse"] <= 0:
        raise ValueError("MAE and RMSE thresholds must be positive.")

    return config