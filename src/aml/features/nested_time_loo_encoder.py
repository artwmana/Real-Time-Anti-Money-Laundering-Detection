from __future__ import annotations

import numpy as np
import pandas as pd 

from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin

class Nested_Time_loo_encoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cols: list[str] | None = None,
                 time_col = None,
                 n_splits = 5,
                 alpha = 50,
                 min_count = 20,
                 use_logit = False,
                 eps = 1e-6,
                 random_state = 42):
        self.cols = cols
        self.time_col = time_col
        self.alpha = alpha
        self.min_count = min_count
        self.use_logit = use_logit
        self.eps = eps
        self.random_state = random_state
        self.n_splits = n_splits

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
            x, rare = self._make_rare(x)
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
            oof = np.full(n, self._default_value(), dtype=float)

            if len(folds) > 0:
                first_train_idx = folds[0][0]
                x_first, rare_first = self._make_rare(x.iloc[first_train_idx])
                first_time = X.iloc[first_train_idx][self.time_col] if self.time_col in X.columns else None
                oof[first_train_idx] = self._causal_loo_stat(x_first, y.iloc[first_train_idx], first_time)

            for train_idx, val_idx in folds:
                X_train, y_train = x.iloc[train_idx].copy(), y.iloc[train_idx]
                X_val = x.iloc[val_idx].copy()

                X_train, rare = self._make_rare(X_train)
                enc = self._stat(X_train, y_train)

                X_val = self._apply_rare(X_val, rare, enc.index)
                oof[val_idx] = self._map_encodings(X_val, enc)
            
            encoded_parts.append(oof)
        
        self.fit(X, y)
        return np.vstack(encoded_parts).T
    
    def transform(self, X):
        if not hasattr(self, "stats_"):
            raise RuntimeError("Encoder is not fitted")
        
        X = pd.DataFrame(X).copy()
        encoded_parts = []

        for col in self.cols:
            x = X[col].copy()
            rare = self.rare_categories_.get(col, set())
            enc = self.stats_[col]

            x = self._apply_rare(x, rare, enc.index)
            encoded_parts.append(self._map_encodings(x, enc))
        
        return np.vstack(encoded_parts).T

    def _apply_rare(self, x: pd.Series, rare_classes: set[str], known_cats: pd.Index) -> pd.Series:
        x = x.astype(str)
        mask = x.isin(rare_classes) | ~x.isin(known_cats)
        x.loc[mask] = "RARE"
        return x
    
    def _make_rare(self, X: pd.Series):
        x = X.astype(str).copy()
        freq = x.value_counts()
        rare_classes = set(freq[freq < self.min_count].index.astype(str))
        x = x.where(~x.isin(rare_classes), "RARE")
        return x, rare_classes

    def _make_folds(self, X: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(X) 
        if n == 0:
            return []

        if self.time_col in X.columns:
            ordered_idx = np.argsort(X[self.time_col].to_numpy())
        else:
            ordered_idx = np.arange(n)

        if n <= self.n_splits:
            return []

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        folds = []

        for train_idx, val_idx in tscv.split(ordered_idx):
            folds.append((ordered_idx[train_idx], ordered_idx[val_idx]))
        return folds
    
    def _stat(self, x: pd.Series, y: pd.Series) -> pd.Series:
        grp = y.groupby(x.to_numpy()).agg(["sum", "count"])
        enc = (grp["sum"] + self.alpha * self._global_mean) / (grp["count"] + self.alpha)
        if self.use_logit:
            enc = self._to_logit(enc)
        return enc
    
    def _causal_loo_stat(self, x: pd.Series, y: pd.Series, time: pd.Series | None = None) -> np.ndarray:
        x = x.astype(str)
        y = pd.Series(y, index=x.index)

        if time is not None:
            order = np.argsort(pd.Series(time).to_numpy(), kind="stable")
            x = x.iloc[order]
            y = y.iloc[order]

        encoded = np.empty(len(x), dtype=float)
        sum_by_cat: dict[str, float] = {}
        count_by_cat: dict[str, int] = {}

        for i, (cat, target) in enumerate(zip(x.to_numpy(), y.to_numpy())):
            prev_sum = sum_by_cat.get(cat, 0.0)
            prev_count = count_by_cat.get(cat, 0)
            value = (prev_sum + self.alpha * self._global_mean) / (prev_count + self.alpha)
            encoded[i] = value
            sum_by_cat[cat] = prev_sum + float(target)
            count_by_cat[cat] = prev_count + 1

        if self.use_logit:
            encoded = self._to_logit(pd.Series(encoded)).to_numpy()

        if time is not None:
            restored = np.empty(len(encoded), dtype=float)
            restored[order] = encoded
            return restored

        return encoded
    
    def _map_encodings(self, x: pd.Series, enc: pd.Series) -> np.ndarray:
        mapped = x.map(enc).fillna(self._default_value())
        return mapped.to_numpy()
    
    def _to_logit(self, s: pd.Series) -> pd.Series:
        clipped = s.clip(self.eps, 1 - self.eps)
        return np.log(clipped / (1 - clipped))

    def _default_value(self) -> float:
        if self.use_logit:
            return float(self._to_logit(pd.Series([self._global_mean])).iloc[0])
        return self._global_mean
