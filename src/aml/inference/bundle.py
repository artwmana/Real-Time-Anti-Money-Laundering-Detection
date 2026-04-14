from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aml.config import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    LEGACY_TARGET_COL,
    MODEL_VERSION,
    POLICY_VERSION,
    SCHEMA_VERSION,
    TARGET_COL,
    get_settings,
)
from aml.models import AMLEnsemble
from aml.persistence.model_persistence import load_base_models_only
from aml.pipelines import FeaturePipeline


@dataclass
class InferenceBundle:
    pipeline: FeaturePipeline
    model: AMLEnsemble
    target_col: str
    schema_version: str
    model_version: str
    policy_version: str
    threshold_review: float
    threshold_block: float
    feature_names: list[str]


def save_inference_bundle(bundle: InferenceBundle, path: str | Path) -> Path:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path, compress=3)
    return save_path


def load_inference_bundle(path: str | Path) -> InferenceBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, InferenceBundle):
        raise TypeError(f"Expected InferenceBundle, got {type(bundle)}")
    return bundle


def _derive_thresholds(model: AMLEnsemble) -> tuple[float, float]:
    learned_threshold = float(getattr(model, "best_stack_threshold", DEFAULT_REVIEW_THRESHOLD) or DEFAULT_REVIEW_THRESHOLD)
    review = max(DEFAULT_REVIEW_THRESHOLD, learned_threshold)
    review = min(max(review, 0.05), 0.95)
    block = max(DEFAULT_BLOCK_THRESHOLD, round(review + 0.15, 4))
    block = min(block, 0.99)
    if block <= review:
        block = min(review + 0.05, 0.99)
    return review, block


def _load_training_frame(settings) -> tuple[pd.DataFrame, str]:
    if settings.processed_train_path.exists():
        df = pd.read_parquet(settings.processed_train_path)
        if TARGET_COL in df.columns:
            return df, TARGET_COL
        if LEGACY_TARGET_COL in df.columns:
            return df, LEGACY_TARGET_COL

    df = pd.read_csv(settings.raw_data_path)
    if TARGET_COL in df.columns:
        return df, TARGET_COL
    if LEGACY_TARGET_COL in df.columns:
        return df, LEGACY_TARGET_COL
    raise KeyError("No supported target column found in training frame")


def build_bundle_from_legacy_artifacts(settings=None) -> InferenceBundle:
    settings = settings or get_settings()
    model = load_base_models_only(settings.legacy_model_dir)
    train_df, target_col = _load_training_frame(settings)
    target = train_df[target_col]

    pipeline = FeaturePipeline(enable_encoding=True, enable_columns=True, alpha=20, use_logit=True)
    pipeline.fit_transform(train_df, target)
    review_threshold, block_threshold = _derive_thresholds(model)

    return InferenceBundle(
        pipeline=pipeline,
        model=model,
        target_col=target_col,
        schema_version=SCHEMA_VERSION,
        model_version=MODEL_VERSION,
        policy_version=POLICY_VERSION,
        threshold_review=review_threshold,
        threshold_block=block_threshold,
        feature_names=list(pipeline.encoded_feature_names_ or []),
    )


def ensure_inference_bundle(settings=None) -> InferenceBundle:
    settings = settings or get_settings()
    if settings.inference_bundle_path.exists():
        return load_inference_bundle(settings.inference_bundle_path)

    bundle = build_bundle_from_legacy_artifacts(settings)
    save_inference_bundle(bundle, settings.inference_bundle_path)
    return bundle
