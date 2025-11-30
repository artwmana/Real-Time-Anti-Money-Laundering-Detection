from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Mean target encoding with logistic smoothing"""

    def __init__(self, smoothing: float = 20.0, min_samples: float = 50.0):
        self.smoothing = smoothing
        self.min_samples = min_samples

        # fitted attributes
        self.global_mean_: float | None = None
        self.mapping_: dict[str, pd.Series] = {}
        self.feature_names_in_: np.ndarray | None = None

    # Fit
    def fit(self, X, y):
        # input normalization
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = X.columns.astype(str).to_numpy()
        else:
            X_df = pd.DataFrame(X)
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X_df.shape[1])], dtype=object)

        y = pd.Series(y).astype(float)

        # global target mean
        self.global_mean_ = float(y.mean())
        self.mapping_ = {}

        # per-column encoding maps
        for col in X_df.columns:
            stats = (
                pd.DataFrame({col: X_df[col], "target": y})
                .groupby(col, observed=True)["target"]
                .agg(["mean", "count"])
            )

            smooth = 1.0 / (1.0 + np.exp(-(stats["count"] - self.min_samples) / self.smoothing))
            enc_values = self.global_mean_ * (1.0 - smooth) + stats["mean"] * smooth

            self.mapping_[str(col)] = enc_values

        return self


    # Transform
    def transform(self, X):
        if self.global_mean_ is None or not self.mapping_:
            raise RuntimeError("SmoothedTargetEncoder is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        encoded = pd.DataFrame(index=X_df.index)

        for col in X_df.columns:
            mapping = self.mapping_.get(str(col))

            if mapping is None:
                # unseen column – fallback to global mean
                encoded[str(col)] = self.global_mean_
            else:
                encoded[str(col)] = X_df[col].map(mapping).fillna(self.global_mean_)

        return encoded.to_numpy(dtype=float)

    # Feature names (CRITICAL FIX)
    def get_feature_names_out(self, input_features=None):
        """
        Required for full compatibility with:
        - ColumnTransformer
        - Pipeline
        - sklearn >= 1.0
        """
        if input_features is not None:
            return np.asarray([str(c) for c in input_features], dtype=object)

        if self.feature_names_in_ is None:
            raise RuntimeError("Encoder is not fitted yet.")

        return self.feature_names_in_
