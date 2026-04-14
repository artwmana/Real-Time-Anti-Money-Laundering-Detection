from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NullMLflowTracker:
    def log_training_run(self, **kwargs: Any) -> None:
        return None


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("mlflow package is required for MLflowTracker") from exc

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        target_col: str,
        split_stats: dict[str, Any],
        test_metrics: dict[str, Any],
        model_dir: Path,
        bundle_path: Path,
        feature_names: list[str],
    ) -> None:
        with self.mlflow.start_run(run_name="aml-training"):
            self.mlflow.log_param("target_col", target_col)
            self.mlflow.log_param("feature_count", len(feature_names))
            self.mlflow.log_metrics({k: float(v) for k, v in split_stats.items() if isinstance(v, (int, float))})
            self.mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

            feature_path = model_dir / "feature_names.json"
            feature_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
            self.mlflow.log_artifact(str(feature_path))
            self.mlflow.log_artifact(str(bundle_path))
