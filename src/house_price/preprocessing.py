from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Missing values in these columns generally indicate that
# the property does not have the corresponding feature.
ABSENCE_CATEGORICAL_FEATURES = {
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
}


def build_preprocessor(X):
    """Build preprocessing for numerical and categorical features."""

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    absence_features = [
        col for col in categorical_features
        if col in ABSENCE_CATEGORICAL_FEATURES
    ]

    regular_categorical_features = [
        col for col in categorical_features
        if col not in ABSENCE_CATEGORICAL_FEATURES
    ]

    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
            ]
        )

        transformers.append(
            ("num", numerical_pipeline, numerical_features)
        )

    if absence_features:
        absence_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="None",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("cat_absence", absence_pipeline, absence_features)
        )

    if regular_categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Missing",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            ("cat", categorical_pipeline, regular_categorical_features)
        )

    if not transformers:
        raise ValueError("No usable features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )