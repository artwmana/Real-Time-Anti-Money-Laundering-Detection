from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import pandas as pd

from aml.models.build_model import AMLEnsemble
from aml.persistence.model_persistence import save_aml_ensemble, save_base_models_only
from aml.pipelines.feature_pipeline import FeaturePipeline
from aml.evaluating.time_split import chronological_split, describe_split, validate_split_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    DATA_PATH = Path(os.getenv("DATA_PATH", "data")).expanduser()
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "models")).expanduser()
    CSV_PATH = DATA_PATH / "raw/AMLNet_August_2025.csv"
    TARGET_COL = "isFraud"
    SPLIT_CFG = dict(test_days=20, val_days=15, gap_days=1)

    logger.info("Starting Optuna training pipeline")
    logger.info("Project root: %s", project_root)
    logger.info("Data path: %s", DATA_PATH)
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("CSV path: %s", CSV_PATH)
    logger.info("Target column: %s", TARGET_COL)
    logger.info("Split config: %s", SPLIT_CFG)

    raw_df = pd.read_csv(CSV_PATH)
    logger.info("Loaded raw dataset with shape=%s", raw_df.shape)
    target = raw_df[TARGET_COL]
    raw_df = FeaturePipeline(enable_columns=True, enable_encoding=False).fit_transform(raw_df, target)
    raw_df[TARGET_COL] = target
    logger.info("Generated pre-split features with shape=%s", raw_df.shape)

    split = chronological_split(raw_df, timestamp_col="timestamp_ts", **SPLIT_CFG)
    validate_split_targets(split, target=TARGET_COL)
    stats = describe_split(split, target=TARGET_COL)
    logger.info("Split stats: %s", stats)

    train_df = split.train.sort_values("timestamp_ts").reset_index(drop=True)
    val_df = split.val.sort_values("timestamp_ts").reset_index(drop=True)
    test_df = split.test.sort_values("timestamp_ts").reset_index(drop=True)

    pipe = FeaturePipeline(enable_encoding=True, enable_columns=True, alpha=20, use_logit=True)
    logger.info("Initialized encoded feature pipeline")

    X_train = pipe.fit_transform(train_df, train_df[TARGET_COL])
    X_val = pipe.transform(val_df)
    X_test = pipe.transform(test_df)
    logger.info("Encoded train shape=%s", X_train.shape)
    logger.info("Encoded val shape=%s", X_val.shape)
    logger.info("Encoded test shape=%s", X_test.shape)

    y_train = train_df[TARGET_COL].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    logger.info(
        "Target positives | train=%d | val=%d | test=%d",
        int(y_train.sum()),
        int(y_val.sum()),
        int(y_test.sum()),
    )

    ensemble = AMLEnsemble(
        n_trials=50,
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
    logger.info("Saved ensemble artifacts to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
