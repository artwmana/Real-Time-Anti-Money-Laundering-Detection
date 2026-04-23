from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import pandas as pd

from aml.config import DEFAULT_BLOCK_THRESHOLD, DEFAULT_REVIEW_THRESHOLD, LEGACY_TARGET_COL, TARGET_COL, get_settings
from aml.infrastructure.mlflow_tracker import MLflowTracker, NullMLflowTracker
from aml.models.build_model import AMLEnsemble
from aml.inference.bundle import InferenceBundle, save_inference_bundle
from aml.persistence.model_persistence import save_aml_ensemble, save_base_models_only
from aml.pipelines.feature_pipeline import FeaturePipeline
from aml.evaluating.time_split import chronological_split, describe_split, validate_split_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    project_root = settings.project_root
    load_dotenv(project_root / ".env")
    DATA_PATH = Path(os.getenv("DATA_PATH", settings.data_path)).expanduser()
    MODEL_PATH = Path(os.getenv("MODEL_PATH", settings.models_path)).expanduser()
    CSV_PATH = DATA_PATH / "raw/AMLNet_August_2025.csv"
    target_col = os.getenv("AML_TARGET_COL", TARGET_COL)
    SPLIT_CFG = dict(test_days=20, val_days=15, gap_days=1)
    N_TRIALS = int(os.getenv("AML_N_TRIALS", "50"))
    SAMPLE_ROWS = int(os.getenv("AML_SAMPLE_ROWS", "0"))

    logger.info("Starting Optuna training pipeline")
    logger.info("Project root: %s", project_root)
    logger.info("Data path: %s", DATA_PATH)
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("CSV path: %s", CSV_PATH)
    logger.info("Target column: %s", target_col)
    logger.info("Split config: %s", SPLIT_CFG)
    logger.info("n_trials: %s", N_TRIALS)
    logger.info("sample_rows: %s", SAMPLE_ROWS)

    read_kwargs = {"nrows": SAMPLE_ROWS} if SAMPLE_ROWS > 0 else {}
    raw_df = pd.read_csv(CSV_PATH, **read_kwargs)
    logger.info("Loaded raw dataset with shape=%s", raw_df.shape)
    if target_col not in raw_df.columns and LEGACY_TARGET_COL in raw_df.columns:
        target_col = LEGACY_TARGET_COL
    target = raw_df[target_col]
    raw_df = FeaturePipeline(enable_columns=True, enable_encoding=False).fit_transform(raw_df, target)
    raw_df[target_col] = target
    logger.info("Generated pre-split features with shape=%s", raw_df.shape)

    split = chronological_split(raw_df, timestamp_col="timestamp_ts", **SPLIT_CFG)
    validate_split_targets(split, target=target_col)
    stats = describe_split(split, target=target_col)
    logger.info("Split stats: %s", stats)

    train_df = split.train.sort_values("timestamp_ts").reset_index(drop=True)
    val_df = split.val.sort_values("timestamp_ts").reset_index(drop=True)
    test_df = split.test.sort_values("timestamp_ts").reset_index(drop=True)

    pipe = FeaturePipeline(enable_encoding=True, enable_columns=True, alpha=20, use_logit=True)
    logger.info("Initialized encoded feature pipeline")

    X_train = pipe.fit_transform(train_df, train_df[target_col])
    X_val = pipe.transform(val_df)
    X_test = pipe.transform(test_df)
    logger.info("Encoded train shape=%s", X_train.shape)
    logger.info("Encoded val shape=%s", X_val.shape)
    logger.info("Encoded test shape=%s", X_test.shape)

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    logger.info(
        "Target positives | train=%d | val=%d | test=%d",
        int(y_train.sum()),
        int(y_val.sum()),
        int(y_test.sum()),
    )

    ensemble = AMLEnsemble(
        n_trials=N_TRIALS,
        meta_splits=5,
        random_state=42,
    )
    logger.info(
        "Initialized ensemble | n_trials=%d | meta_splits=%d | random_state=%d",
        ensemble.n_trials,
        ensemble.meta_splits,
        ensemble.random_state,
    )
    logger.info("Starting ensemble fit")
    ensemble.fit(X_train, y_train, X_val, y_val)
    logger.info("Finished ensemble fit")

    test_metrics = ensemble.evaluate(X_test, y_test)
    logger.info("Test metrics: %s", test_metrics)

    save_aml_ensemble(ensemble, MODEL_PATH / "aml_ensemble")
    save_base_models_only(ensemble, MODEL_PATH / "base_models_only")

    review_threshold = max(DEFAULT_REVIEW_THRESHOLD, float(ensemble.best_stack_threshold))
    block_threshold = min(max(review_threshold + 0.15, DEFAULT_BLOCK_THRESHOLD), 0.99)
    bundle = InferenceBundle(
        pipeline=pipe,
        model=ensemble,
        target_col=target_col,
        schema_version="2026-04-11",
        model_version="aml_ensemble_v1",
        policy_version="aml_policy_v1",
        threshold_review=review_threshold,
        threshold_block=block_threshold,
        feature_names=list(pipe.encoded_feature_names_ or []),
    )
    bundle_path = save_inference_bundle(bundle, MODEL_PATH / "inference_bundle.joblib")
    tracker = NullMLflowTracker()
    if settings.enable_mlflow:
        try:
            tracker = MLflowTracker(settings.mlflow_tracking_uri, settings.mlflow_experiment)
        except Exception:
            logger.exception("MLflow tracker initialization failed; continuing without MLflow")
    tracker.log_training_run(
        target_col=target_col,
        split_stats=stats,
        test_metrics=test_metrics,
        model_dir=MODEL_PATH,
        bundle_path=bundle_path,
        feature_names=list(pipe.encoded_feature_names_ or []),
    )
    logger.info("Saved ensemble artifacts to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
