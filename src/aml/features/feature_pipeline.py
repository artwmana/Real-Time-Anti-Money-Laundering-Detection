from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from aml.features.json_extractor import flatten_metadata
from aml.features.dtype_downcasting import optimize_dataframe
from aml.features.encoding import SmoothedTargetEncoder


@dataclass(frozen=True)
class EncoderConfig:
    low_card_threshold: int = 8
    target_cols: Tuple[str, ...] = ("isFraud", "isMoneyLaundering")


class FeaturePipeline:
    """
    Feature engineering pipeline for offline training and real-time inference.

    Stages:
    - JSON flattening
    - dtype optimization
    - feature engineering
    - encoding (ColumnTransformer)
    - final cleanup
    """

    def __init__(
        self,
        enable_downcasting: bool = True,
        enable_encoding: bool = True,
        enable_scale: bool = False,
        drop_original_metadata: bool = True,
        encoder_cfg: EncoderConfig = EncoderConfig(),
    ):
        self.enable_downcasting = enable_downcasting
        self.enable_encoding = enable_encoding
        self._enable_scale = enable_scale
        self.drop_original_metadata = drop_original_metadata
        self.encoder_cfg = encoder_cfg

        # Will be set during fit
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_columns_: Optional[list[str]] = None  # optional: for debugging/tracking

    @property
    def enable_scale(self) -> bool:
        return bool(self._enable_scale and self.enable_encoding)

    def fit_transform(self, df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Training path:
        - build features
        - fit encoder
        - transform train set
        """
        self._validate_input(df)
        feats = self._build_features(df)

        # store feature col names for debugging
        self.feature_columns_ = feats.columns.tolist()

        if not self.enable_encoding:
            # If you really want DataFrame out when encoding disabled, return feats.values
            return feats.to_numpy()

        return self._fit_encoder(feats, y_train)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Inference path:
        - build features
        - transform via fitted encoder
        """
        self._validate_input(df)
        feats = self._build_features(df)

        if not self.enable_encoding:
            return feats.to_numpy()

        if self.preprocessor is None:
            raise RuntimeError("Encoder is not fitted. Call fit_transform() first.")

        X = self._drop_targets(feats)
        return self.preprocessor.transform(X)

    def transform_single(self, record: Dict[str, Any]) -> np.ndarray:
        return self.transform(pd.DataFrame([record]))

    # -------------------------------
    # Core pipeline
    # -------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Flatten JSON metadata (if present)
        if "metadata" in df.columns:
            df = flatten_metadata(df)
            if self.drop_original_metadata:
                df = df.drop(columns=["metadata"], errors="ignore")

        # 2. Basic time normalization early (if needed later)
        df = self._ensure_time_columns(df)

        # 3) Optimize dtypes
        if self.enable_downcasting:
            df = optimize_dataframe(df)

        # 4. Domain feature engineering
        df = self._feature_engineering(df)

        # 5. Final cleanup (NaNs, inf, dtypes)
        df = self._finalize(df)

        return df

    # -------------------------------
    # Feature engineering
    # -------------------------------

    def _ensure_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # If timestamp exists, ensure datetime + derive hour/day_of_week if missing
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            if "hour" not in df.columns:
                df["hour"] = df["timestamp"].dt.hour

            if "day_of_week" not in df.columns:
                df["day_of_week"] = df["timestamp"].dt.dayofweek

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- Time based ----
        if "timestamp" in df.columns:
            df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype("int8")

        if "hour" in df.columns:
            # handle NaNs safely
            h = df["hour"].fillna(0).astype(float)
            df["hour_sin"] = np.sin(2 * np.pi * h / 24)
            df["hour_cos"] = np.cos(2 * np.pi * h / 24)

        if "day_of_week" in df.columns:
            d = df["day_of_week"].fillna(0).astype(float)
            df["dow_sin"] = np.sin(2 * np.pi * d / 7)
            df["dow_cos"] = np.cos(2 * np.pi * d / 7)

        # ---- Balance deltas and ratios ----
        if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
            df["balance_delta"] = df["oldbalanceOrg"].astype(float) - df["newbalanceOrig"].astype(float)

        if {"amount", "oldbalanceOrg"}.issubset(df.columns):
            denom = df["oldbalanceOrg"].replace({0: np.nan}).astype(float)
            df["amt_to_balance"] = (df["amount"].astype(float) / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

        # ---- User aggregations (requires nameOrig) ----
        if "nameOrig" in df.columns:
            user_grp = df.groupby("nameOrig", observed=True)

            if "step" in df.columns:
                df["user_tx_count"] = user_grp["step"].transform("count").astype("int32")

            if "amount" in df.columns:
                df["user_amount_median"] = user_grp["amount"].transform("median").astype(float)
                df["user_amount_std"] = user_grp["amount"].transform("std").fillna(0).astype(float)
                df["amt_to_user_median"] = df["amount"].astype(float) / (df["user_amount_median"].astype(float) + 1e-3)

            if "nameDest" in df.columns:
                df["user_unique_merchants"] = user_grp["nameDest"].transform("nunique").astype("int32")

            if "device_type" in df.columns:
                df["user_device_diversity"] = user_grp["device_type"].transform("nunique").astype("int16")

        # ---- Merchant / device level ----
        if "merch_merchant_id" in df.columns and "amount" in df.columns:
            merchant_grp = df.groupby("merch_merchant_id", observed=True)
            df["merchant_tx_count"] = merchant_grp["amount"].transform("count").astype("int32")
            df["amt_to_merchant_avg"] = df["amount"].astype(float) / (merchant_grp["amount"].transform("mean").astype(float) + 1e-3)

        if "device_ip_address" in df.columns and "step" in df.columns:
            ip_grp = df.groupby("device_ip_address", observed=True)
            df["ip_activity_rank"] = ip_grp["step"].transform("count").astype("int32")

        # ---- Risk-based ----
        if {"risk_risk_score", "risk_customer_risk_score"}.issubset(df.columns):
            df["risk_score_gap"] = df["risk_risk_score"].astype(float) - df["risk_customer_risk_score"].astype(float)

        if "risk_amount_vs_average" in df.columns:
            df["amount_vs_avg_diff"] = df["risk_amount_vs_average"].astype(float) - 1.0

        if "risk_customer_risk_score" in df.columns:
            base = df["risk_customer_risk_score"].astype(float)
            med = float(np.nanmedian(base.to_numpy())) if np.isfinite(base).any() else 0.5
            df["customer_risk_bucket"] = pd.cut(
                base.fillna(med),
                bins=[-np.inf, 0.3, 0.6, np.inf],
                labels=["low", "medium", "high"],
            ).astype("category")

        return df

    # -------------------------------
    # Encoding
    # -------------------------------

    def _drop_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=list(self.encoder_cfg.target_cols), errors="ignore")

    def _fit_encoder(self, df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        X = self._drop_targets(df)

        # safety: drop datetime columns to avoid numpy dtype promotion error
        dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        if len(dt_cols) > 0:
            X = X.drop(columns=list(dt_cols))


        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        low_card_cols = [c for c in categorical_cols if X[c].nunique(dropna=True) <= self.encoder_cfg.low_card_threshold]
        high_card_cols = [c for c in categorical_cols if c not in low_card_cols]
        numeric_cols = X.columns.difference(categorical_cols).tolist()

        # Numeric pipeline
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.enable_scale:
            num_steps.append(("scaler", RobustScaler()))
        num_pipe = Pipeline(num_steps)

        # Low cardinality: OneHot
        low_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # High cardinality: Target encoding (time to be careful with leakage in validation!)
        high_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target", SmoothedTargetEncoder()),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("low_card", low_card_pipe, low_card_cols),
                ("high_card", high_card_pipe, high_card_cols),
            ],
            remainder="drop",
        )

        self.preprocessor.fit(X, y_train)
        return self.preprocessor.transform(X)

    # -------------------------------
    # Final cleanup
    # -------------------------------

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # convert timestamp -> unix seconds (numeric feature) and drop datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            df["timestamp_ts"] = (df["timestamp"].astype("int64") // 10**9).astype("float64")
            df.loc[df["timestamp"].isna(), "timestamp_ts"] = np.nan

            df.drop(columns=["timestamp"], inplace=True)

        # drop ANY remaining datetime columns just in case
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        if len(dt_cols) > 0:
            df = df.drop(columns=list(dt_cols))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)

        return df

