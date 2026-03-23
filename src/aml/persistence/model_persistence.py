from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

import joblib

from aml.models import AMLEnsemble


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_from_ensemble(model: AMLEnsemble) -> Dict[str, Any]:
    return {
        "class_name": model.__class__.__name__,
        "n_trials": model.n_trials,
        "meta_splits": model.meta_splits,
        "random_state": model.random_state,
        "early_stopping_rounds": model.early_stopping_rounds,
        "oof_early_stopping_rounds": model.oof_early_stopping_rounds,
        "scale_pos_weight": model.scale_pos_weight,
        "model_names": list(model.model_names),
        "best_params": model.best_params,
        "best_iterations": model.best_iterations,
        "best_stack_threshold": model.best_stack_threshold,
        "is_fitted": bool(model.meta_model is not None and bool(model.base_models_trainval)),
    }


def save_aml_ensemble(model: AMLEnsemble, save_dir: str | Path) -> Path:
    if not isinstance(model, AMLEnsemble):
        raise TypeError("model must be an instance of AMLEnsemble")

    save_path = _ensure_dir(save_dir)

    metadata = _metadata_from_ensemble(model)
    with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    joblib.dump(model, save_path / "ensemble.joblib", compress=3)
    return save_path


def load_aml_ensemble(save_dir: str | Path) -> AMLEnsemble:
    save_path = Path(save_dir)
    model_path = save_path / "ensemble.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved ensemble not found: {model_path}")

    model = joblib.load(model_path)
    if not isinstance(model, AMLEnsemble):
        raise TypeError(f"Loaded object is not AMLEnsemble, got: {type(model)}")
    return model


def save_base_models_only(model: AMLEnsemble, save_dir: str | Path) -> Path:
    if model.meta_model is None or not model.base_models_trainval:
        raise ValueError("Model must be fitted before saving deployment artifacts.")

    save_path = _ensure_dir(save_dir)

    artifacts = {
        "base_models_trainval": model.base_models_trainval,
        "meta_model": model.meta_model,
        "best_stack_threshold": model.best_stack_threshold,
        "best_params": model.best_params,
        "best_iterations": model.best_iterations,
        "scale_pos_weight": model.scale_pos_weight,
        "model_names": model.model_names,
        "config": {
            "n_trials": model.n_trials,
            "meta_splits": model.meta_splits,
            "random_state": model.random_state,
            "early_stopping_rounds": model.early_stopping_rounds,
            "oof_early_stopping_rounds": model.oof_early_stopping_rounds,
        },
    }

    joblib.dump(artifacts, save_path / "ensemble_deploy.joblib", compress=3)

    with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **_metadata_from_ensemble(model),
                "artifact_type": "deployment_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return save_path


def load_base_models_only(save_dir: str | Path) -> AMLEnsemble:
    save_path = Path(save_dir)
    artifact_path = save_path / "ensemble_deploy.joblib"

    if not artifact_path.exists():
        raise FileNotFoundError(f"Deployment artifact not found: {artifact_path}")

    artifacts = joblib.load(artifact_path)

    config = artifacts["config"]
    model = AMLEnsemble(
        n_trials=config["n_trials"],
        meta_splits=config["meta_splits"],
        random_state=config["random_state"],
        early_stopping_rounds=config["early_stopping_rounds"],
        oof_early_stopping_rounds=config["oof_early_stopping_rounds"],
    )

    model.base_models_trainval = artifacts["base_models_trainval"]
    model.meta_model = artifacts["meta_model"]
    model.best_stack_threshold = artifacts["best_stack_threshold"]
    model.best_params = artifacts["best_params"]
    model.best_iterations = artifacts["best_iterations"]
    model.scale_pos_weight = artifacts["scale_pos_weight"]
    model.model_names = tuple(artifacts["model_names"])

    return model
