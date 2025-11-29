from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    '''Mean target encoding with logistic smoothing'''
    
    def __init__(self, smoothing: float = 20.0, min_samples: float = 50.0):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).astype(float)
        self.global_mean_ = y.mean()
        for col in X.columns:
            stats = (
                pd.DataFrame({col: X[col], 'target': y})
                .groupby(col)['target']
                .agg(['mean', 'count'])
            )
            smooth = 1 / (1 + np.exp(-(stats['count'] - self.min_samples) / self.smoothing))
            enc_values = self.global_mean_ * (1 - smooth) + stats['mean'] * smooth
            self.mapping_[col] = enc_values
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        encoded = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mapping_.get(col, {})
            encoded[col] = X[col].map(mapping).fillna(self.global_mean_)
        return encoded.values