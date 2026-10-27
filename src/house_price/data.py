from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TARGET_COLUMN = "SalePrice"


def validate_raw_data(df: pd.DataFrame) -> None:
    """Check whether the raw dataset is suitable for model training."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    if len(df.columns) < 2:
        raise ValueError("Dataset must contain features and a target.")

    target = df[TARGET_COLUMN]

    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain numeric values."
        )

    if target.isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains missing values."
        )

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains infinite or invalid values."
        )

    if (target <= 0).any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains non-positive prices."
        )


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original CSV and validate its essential structure."""

    data_path = Path(path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Place train.csv inside data/raw/."
        )

    df = pd.read_csv(data_path)
    validate_raw_data(df)

    return df