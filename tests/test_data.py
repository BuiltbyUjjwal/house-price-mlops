import pandas as pd
import pytest

from house_price.data import validate_raw_data


def valid_dataframe():
    """Create a small valid dataset for testing."""

    return pd.DataFrame({
        "LotArea": [8000, 10000, 12000],
        "Neighborhood": ["A", "B", "A"],
        "SalePrice": [150000, 200000, 250000],
    })


def test_valid_data_passes():
    """Valid data should pass validation."""

    df = valid_dataframe()

    validate_raw_data(df)


def test_missing_target_column_fails():
    """Data without SalePrice should be rejected."""

    df = valid_dataframe().drop(columns=["SalePrice"])

    with pytest.raises(ValueError, match="target column"):
        validate_raw_data(df)


def test_missing_target_value_fails():
    """Rows without a target value should be rejected."""

    df = valid_dataframe()
    df.loc[0, "SalePrice"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_raw_data(df)


def test_non_positive_price_fails():
    """House prices must be positive."""

    df = valid_dataframe()
    df.loc[0, "SalePrice"] = 0

    with pytest.raises(ValueError, match="non-positive"):
        validate_raw_data(df)


def test_infinite_price_fails():
    """Infinite target values should be rejected."""

    df = valid_dataframe()

    # Convert the column to float before assigning infinity.
    df["SalePrice"] = df["SalePrice"].astype("float64")
    df.loc[0, "SalePrice"] = float("inf")

    with pytest.raises(ValueError, match="infinite"):
        validate_raw_data(df)


def test_duplicate_column_names_fail():
    """Duplicate column names should be rejected."""

    df = valid_dataframe()
    df.columns = ["LotArea", "SalePrice", "SalePrice"]

    with pytest.raises(ValueError, match="duplicate column"):
        validate_raw_data(df)