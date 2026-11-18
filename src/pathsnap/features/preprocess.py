
"""Training-only feature imputation and standardization."""

from pathlib import Path
from typing import Self

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pathsnap.features.build import FEATURE_COLUMNS


class FeaturePreprocessor:
    """Fit feature transformations on training data and reuse them."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.is_fitted = False

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Check that all required feature columns are present."""
        missing = set(self.feature_columns) - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing required feature columns: {sorted(missing)}"
            )

    def fit(self, data: pd.DataFrame) -> Self:
        """Fit imputation and scaling using training rows only."""

        self._validate_columns(data)

        if data.empty:
            raise ValueError("Cannot fit preprocessing on empty data")

        if "partition" not in data.columns:
            raise ValueError("Data must contain a partition column")

        partitions = set(data["partition"].dropna().unique())

        if partitions != {"train"}:
            raise ValueError(
                "Preprocessor fit is allowed on training data only; "
                f"received partitions: {partitions}"
            )

        if data[self.feature_columns].isna().all().any():
            raise ValueError(
                "At least one feature is entirely missing in training data"
            )

        values = data[self.feature_columns].astype(float)

        # Learn median values from training data.
        imputed = self.imputer.fit_transform(values)

        # Learn scaling parameters from training data.
        self.scaler.fit(imputed)

        self.is_fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations without refitting."""

        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )

        self._validate_columns(data)

        result = data.copy()

        if result.empty:
            return result

        # Convert features to floating-point values.
        values = result[self.feature_columns].astype(float)

        # Apply training-fitted imputation.
        imputed = self.imputer.transform(values)

        # Apply training-fitted standardization.
        scaled = self.scaler.transform(imputed)

        # Store transformed values without changing metadata columns.
        result[self.feature_columns] = scaled

        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed features."""

        self.fit(data)

        return self.transform(data)

    def save(self, path: str | Path) -> None:
        """Save the fitted preprocessing state."""

        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted preprocessor"
            )

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self, output_path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously saved fitted preprocessor."""

        loaded = joblib.load(Path(path))

        if not isinstance(loaded, cls):
            raise TypeError(
                "Saved artifact is not a FeaturePreprocessor"
            )

        if not loaded.is_fitted:
            raise RuntimeError(
                "Saved preprocessor is not fitted"
            )

        return loaded
