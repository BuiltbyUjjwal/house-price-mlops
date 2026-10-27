import pandas as pd

from house_price.data import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    load_raw_data,
)


def main() -> None:
    """Print a useful profile of the raw house-price dataset."""

    try:
        df = load_raw_data()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"DATA CHECK FAILED: {exc}") from exc

    print("\n========== DATASET OVERVIEW ==========")
    print(f"File: {RAW_DATA_PATH}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    if "Id" in df.columns:
        print(f"Duplicate IDs: {df['Id'].duplicated().sum()}")

    print("\n========== FIRST FIVE ROWS ==========")
    print(df.head().to_string(max_cols=10))

    print("\n========== COLUMN TYPES ==========")
    print(df.dtypes.astype(str).value_counts().to_string())

    numerical_features = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != TARGET_COLUMN
    ]

    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"\nNumerical feature columns: {len(numerical_features)}")
    print(f"Categorical feature columns: {len(categorical_features)}")

    print("\n========== TARGET STATISTICS ==========")
    print(
        df[TARGET_COLUMN]
        .describe(percentiles=[0.25, 0.50, 0.75, 0.90])
        .round(2)
        .to_string()
    )

    print("\n========== MISSING VALUES ==========")

    missing_report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    )

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ].sort_values("missing_percent", ascending=False)

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.head(20).to_string())

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    main()