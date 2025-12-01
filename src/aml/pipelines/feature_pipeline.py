from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable
from functools import wraps
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from aml.features.json_extractor import flatten_metadata
from aml.features.dtype_downcasting import optimize_dataframe
from aml.features.kfold_target_encoder import TimeKFoldTargetEncoder

logger = logging.getLogger(__name__)

# Decorator for stage logs
def _shape_of(x: Any):
    if isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)):
        return x.shape
    return None

def log_stage(_fn: Optional[Callable] = None, *, name: Optional[str] = None, level: int = logging.INFO):
    def decorator(fn: Callable):
        stage = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            lg = getattr(args[0], "logger", logger) if args else logger

            in_shape = None
            for v in list(args[1:]) + list(kwargs.values()):
                s = _shape_of(v)
                if s is not None:
                    in_shape = s
                    break

            t0 = perf_counter()
            lg.log(level, "▶ %s | in_shape=%s", stage, in_shape)

            try:
                out = fn(*args, **kwargs)
            except Exception:
                lg.exception("✖ %s failed", stage)
                raise

            out_shape = _shape_of(out[0] if isinstance(out, tuple) else out)
            lg.log(level, "✓ %s | out_shape=%s | %.3fs", stage, out_shape, perf_counter() - t0)
            return out

        return wrapper

    return decorator if _fn is None else decorator(_fn)

# Config
@dataclass(frozen=True)
class EncoderConfig:
    low_card_threshold: int = 8
    target_cols: Tuple[str, ...] = ("isFraud", "isMoneyLaundering")


# Feature Pipeline
class FeaturePipeline:
    """
    Feature Pipeline for both offline and real-time features

    Stages:
    - Check input
    - Flatten JSON
    - Optimize dtypes
    - Adding new features (feature engineering)
    - Encoding
    """

    def __init__(
        self,
        enable_downcasting: bool = True,
        enable_encoding: bool = True,
        enable_scale: bool = False,
        enable_columns: bool = False,
        drop_original_metadata: bool = True,
        encoder_cfg: EncoderConfig = EncoderConfig(),
    ):
        self.enable_downcasting = enable_downcasting
        self.enable_encoding = enable_encoding
        self._enable_scale = enable_scale
        self.enable_columns = enable_columns
        self.drop_original_metadata = drop_original_metadata
        self.encoder_cfg = encoder_cfg

        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_columns_: Optional[list[str]] = None
        self.encoded_feature_names_: Optional[list[str]] = None
        self.numeric_cols_: list[str] = []
        self.low_card_cols_: list[str] = []
        self.ordinal_cols_: list[str] = []
        self.high_card_cols_: list[str] = []

        logger.info(
            "FeaturePipeline initialized | downcasting=%s | encoding=%s | _scale=%s | columns=%s",
            enable_downcasting, enable_encoding, enable_scale, enable_columns
        )

        if enable_scale and not enable_encoding:
            logger.warning("Config: enable_scale=True but enable_encoding=False → scale disabled")

    @property
    def enable_scale(self) -> bool:
        return bool(self._enable_scale and self.enable_encoding)

    # For public API
    @log_stage(name="fit_transform")
    def fit_transform(self, df: pd.DataFrame, y_train: pd.Series):
        self._validate_input(df)
        feats = self._build_features(df)

        if not self.enable_encoding:
            return feats if not self.enable_columns else self._adding_col(feats)

        Xt = self._fit_encoder(feats, y_train)

        if self.enable_columns:
            return self._adding_col(Xt)

        return Xt

    @log_stage(name="transform")
    def transform(self, df: pd.DataFrame):
        self._validate_input(df)
        feats = self._build_features(df)

        if not self.enable_encoding:
            return feats.to_numpy()

        if self.preprocessor is None:
            raise RuntimeError("Encoder is not fitted.")

        X = self._drop_targets(feats)
        Xt = self.preprocessor.transform(X)

        if self.enable_columns:
            return self._adding_col(Xt)

        return Xt

    def transform_single(self, record: Dict[str, Any]):
        return self.transform(pd.DataFrame([record]))

    # Core stages
    def _validate_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        self.feature_columns_ = df.columns.tolist()
        logger.info("Input validated | shape=%s", df.shape)

    @log_stage(name="build_features")
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "metadata" in df.columns:
            df = flatten_metadata(df)
            if self.drop_original_metadata:
                df = df.drop(columns=["metadata"], errors="ignore")

        df = self._ensure_time_columns(df)

        if self.enable_downcasting:
            df = optimize_dataframe(df)

        df = self._feature_engineering(df)
        df = self._finalize(df)

        return df

    def _ensure_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            if "hour" not in df.columns:
                df["hour"] = df["timestamp"].dt.hour

            if "day_of_week" not in df.columns:
                df["day_of_week"] = df["timestamp"].dt.dayofweek

        return df

    @log_stage(name="feature_engineering")
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype("int8")

        if "hour" in df.columns:
            h = df["hour"].fillna(0).astype(float)
            df["hour_sin"] = np.sin(2 * np.pi * h / 24)
            df["hour_cos"] = np.cos(2 * np.pi * h / 24)

        if "day_of_week" in df.columns:
            d = df["day_of_week"].fillna(0).astype(float)
            df["dow_sin"] = np.sin(2 * np.pi * d / 7)
            df["dow_cos"] = np.cos(2 * np.pi * d / 7)

        if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
            df["balance_delta"] = df["oldbalanceOrg"].astype(float) - df["newbalanceOrig"].astype(float)

        if {"amount", "oldbalanceOrg"}.issubset(df.columns):
            denom = df["oldbalanceOrg"].replace({0: np.nan}).astype(float)
            df["amt_to_balance"] = (df["amount"].astype(float) / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    def _drop_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=list(self.encoder_cfg.target_cols), errors="ignore")

    @log_stage(name="fit_encoder")
    def _fit_encoder(self, df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        X = self._drop_targets(df)

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.low_card_cols_ = [c for c in categorical_cols if X[c].nunique() <= self.encoder_cfg.low_card_threshold]
        self.ordinal_cols_ = [i for i in self.low_card_cols_ if 'risk' in i]
        self.low_card_cols_ = [c for c in self.low_card_cols_ if c not in self.ordinal_cols_]
        self.high_card_cols_ = [c for c in categorical_cols if c not in self.low_card_cols_]
        self.numeric_cols_ = X.columns.difference(categorical_cols).tolist()

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.enable_scale:
            num_steps.append(("scaler", RobustScaler()))
        num_pipe = Pipeline(num_steps)

        low_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        original_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(min_frequency=50, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        time_col = "timestamp_ts" if "timestamp_ts" in X.columns else None
        if len(self.high_card_cols_) > 0:
            high_card_features = self.high_card_cols_ + ([time_col] if time_col else [])
            high_card_pipe = Pipeline([
                ("target", TimeKFoldTargetEncoder(
                    cols=self.high_card_cols_,
                    time_col=time_col,
                    n_splits=5,
                )),
            ])
        else:
            high_card_features = []
            high_card_pipe = "drop"

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols_),
                ("low_card", low_card_pipe, self.low_card_cols_),
                ("orig", original_pipe, self.ordinal_cols_),
                ("high_card", high_card_pipe, high_card_features),
            ],
            remainder="drop",
        )

        self.preprocessor.fit(X, y_train)
        Xt = self.preprocessor.transform(X)

        self.encoded_feature_names_ = self._get_encoded_feature_names()
        return Xt


    # Finalization
    @log_stage(name="finalize")
    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp_ts"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9
            df.drop(columns=["timestamp"], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)

        return df


    # Column names
    def _get_encoded_feature_names(self) -> list[str]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not fitted")

        if hasattr(self.preprocessor, "get_feature_names_out"):
            try:
                return list(map(str, self.preprocessor.get_feature_names_out()))
            except Exception:
                logger.warning("get_feature_names_out() exists but failed, fallback to manual build")

        names: list[str] = []

        # numeric 
        names += [f"num__{c}" for c in self.numeric_cols_]

        # low-card OneHot
        if len(self.low_card_cols_) > 0:
            low_card_tf = self.preprocessor.named_transformers_.get("low_card")
            if low_card_tf != "drop" and low_card_tf is not None:
                ohe = low_card_tf.named_steps["encoder"]
                try:
                    ohe_names = ohe.get_feature_names_out(self.low_card_cols_)
                    names += [f"low_card__{n}" for n in ohe_names]
                except Exception:
                    for col, cats in zip(self.low_card_cols_, ohe.categories_):
                        names += [f"low_card__{col}_{c}" for c in cats]

        # original
        if len(self.ordinal_cols_) > 0:
            names += [f"orig__{c}" for c in self.ordinal_cols_]

        # high-card (target encoding = 1 колонка на фичу)
        high_card_tf = self.preprocessor.named_transformers_.get("high_card")
        if len(self.high_card_cols_) > 0 and high_card_tf != "drop" and high_card_tf is not None:
            has_count = getattr(high_card_tf.named_steps["target"], "add_count_features", False)
            suffixes = ["enc", "cnt"] if has_count else ["enc"]
            for col in self.high_card_cols_:
                for suf in suffixes:
                    names.append(f"high_card__{col}__{suf}")

        return names



    # Output formatting
    def _adding_col(self, feats):
        if self.enable_encoding and isinstance(feats, np.ndarray):
            cols = self.encoded_feature_names_
            return pd.DataFrame(feats, columns=cols)

        if isinstance(feats, pd.DataFrame):
            return feats
        
        cols = getattr(self, "encoded_feature_names_", None)
        if cols is None:
            raise ValueError("Encoded feature names are not available.")

        if feats.shape[1] != len(cols):
            raise ValueError(
                f"Mismatch between encoded data and feature names: "
                f"X has {feats.shape[1]} columns but got {len(cols)} names"
            )


        return pd.DataFrame(feats)
