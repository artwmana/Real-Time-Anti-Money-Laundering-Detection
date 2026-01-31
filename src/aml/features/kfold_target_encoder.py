from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeKFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with time-aware folds to reduce leakage.

    - During fit_transform we do out-of-fold encoding based on chronological blocks.
    - During transform we apply full-data statistics with smoothing.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        time_col: str | None = None,
        n_splits: int = 5,
        alpfa: float = 200.0,
        min_count: int = 20,
        use_logit: bool = False,
        eps: float = 1e-6,
        random_state: int = 42,
    ):
        self.cols = cols
        self.time_col = time_col
        self.n_splits = n_splits
        self.alpfa = alpfa
        self.min_count = min_count
        self.use_logit = use_logit
        self.eps = eps
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y)

        if self.cols is None:
            self.cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self._global_mean = float(y.mean())
        self.stats_: dict[str, pd.Series] = {}
        self.rare_categories_: dict[str, set[str]] = {}

        for col in self.cols:
            x = X[col].copy()
            x, rare = self._mark_rare(x)
            enc = self._stat(x, y)
            self.stats_[col] = enc
            self.rare_categories_[col] = rare

        return self

    def fit_transform(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y)

        if self.cols is None:
            self.cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self._global_mean = float(y.mean())
        folds = self._make_folds(X)
        n = len(X)
        encoded_parts = []

        for col in self.cols:
            x = X[col].copy()
            oof = np.zeros(n, dtype=float)

            for fold in folds:
                val_idx = fold
                train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=True)

                x_train, y_train = x.iloc[train_idx].copy(), y.iloc[train_idx]
                x_val = x.iloc[val_idx].copy()

                x_train, rare = self._mark_rare(x_train)
                enc = self._stat(x_train, y_train)

                x_val = self._apply_rare(x_val, rare, enc.index)
                oof[val_idx] = self._map_encodings(x_val, enc)

            encoded_parts.append(oof)

        self.fit(X, y)

        return np.vstack(encoded_parts).T

    def transform(self, X):
        if not hasattr(self, "stats_"):
            raise RuntimeError("Encoder is not fitted.")

        X = pd.DataFrame(X).copy()
        encoded_parts = []

        for col in self.cols:
            x = X[col].copy()
            rare = self.rare_categories_.get(col, set())
            enc = self.stats_[col]

            x = self._apply_rare(x, rare, enc.index)
            encoded_parts.append(self._map_encodings(x, enc))

        return np.vstack(encoded_parts).T

    # Helpers
    def _make_folds(self, X: pd.DataFrame) -> list[np.ndarray]:
        n = len(X)
        if n < self.n_splits:
            return [np.arange(n)]

        if self.time_col and self.time_col in X.columns:
            ordered_idx = np.argsort(X[self.time_col].to_numpy())
            folds = np.array_split(ordered_idx, self.n_splits)
        else:
            rng = np.random.default_rng(self.random_state)
            ordered_idx = rng.permutation(n)
            folds = np.array_split(ordered_idx, self.n_splits)

        return [f for f in folds if len(f) > 0]

    def _stat(self, x: pd.Series, y: pd.Series) -> pd.Series:
        x = x.astype(str)
        y = pd.Series(y)

        grp = y.groupby(x).agg(["sum", "count"])
        enc = (grp["sum"] + self.alpfa * self._global_mean) / (grp["count"] + self.alpfa)
        if self.use_logit:
            enc = self._to_logit(enc)
        return enc

    def _mark_rare(self, x: pd.Series) -> tuple[pd.Series, set[str]]:
        freq = x.value_counts()
        rare_classes = set(freq[freq < self.min_count].index.astype(str))
        x = x.astype(str)
        x = x.where(~x.isin(rare_classes), "RARE")
        return x, rare_classes

    def _apply_rare(self, x: pd.Series, rare_classes: set[str], known_cats: pd.Index) -> pd.Series:
        x = x.astype(str)
        mask = x.isin(rare_classes) | ~x.isin(known_cats)
        x.loc[mask] = "RARE"
        return x

    def _map_encodings(self, x: pd.Series, enc: pd.Series) -> np.ndarray:
        mapped = x.map(enc).fillna(self._global_mean)
        return mapped.to_numpy()

    def _to_logit(self, s: pd.Series) -> pd.Series:
        clipped = s.clip(self.eps, 1 - self.eps)
        return np.log(clipped / (1 - clipped))
