from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

class AMLEnsemble:
    def __init__(
        self,
        n_trials: int = 50,
        meta_splits: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        oof_early_stopping_rounds: int = 30,
    ) -> None:
        self.n_trials = n_trials
        self.meta_splits = meta_splits
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.oof_early_stopping_rounds = oof_early_stopping_rounds

        self.scale_pos_weight: float = 1.0
        self.model_names = ("xgb", "lgbm", "cat")

        self.studies: Dict[str, optuna.Study] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.best_iterations: Dict[str, int] = {}

        self.base_models_train: Dict[str, Any] = {}
        self.base_models_trainval: Dict[str, Any] = {}
        self.meta_model: Optional[SkPipeline] = None
        self.best_stack_threshold: float = 0.5

        self.X_train_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[np.ndarray] = None
        self.X_val_: Optional[pd.DataFrame] = None
        self.y_val_: Optional[np.ndarray] = None
        self.X_trainval_: Optional[pd.DataFrame] = None
        self.y_trainval_: Optional[np.ndarray] = None

    def fit_model(self,
                  name: str,
                  model: Any,
                  X_tr: pd.DataFrame,
                  y_tr: np.ndarray,
                  X_va: pd.DataFrame,
                  y_va: np.ndarray,
                  ) -> Any:
            if name == "xgb":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                return model

            if name == "lgbm":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )
                return model

            if name == "cat":
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
                return model

            raise ValueError(f"Unknown model: {name}")

    def tune_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        self._calculate_scale_pos_weight(y_train)
        
        for name in self.model_names:
            print(f"Tuning {name}")
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=self.random_state),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
                study_name=f"{name}_study",
            )
            study.optimize(self._make_objective(name, X_train, y_train, X_val, y_val), 
                            n_trials=self.n_trials,
                            show_progress_bar=False)
            
            self.studies[name] = study
            self.best_params[name] = study.best_params
            print(f"{name} best PR-AUC on val: {study.best_value:.6f}")

        return self.best_params

    def _make_objective(self, name: str, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        def objective(trial: optuna.Trial) -> float:
            params = self.suggest_params(trial, name)
            model = self.build_model(name=name, params=params, early_stopping_rounds=self.early_stopping_rounds)
            model = self.fit_model(name, model, X_train, y_train, X_val, y_val)
            val_proba = model.predict_proba(X_val)[:, 1]
            return self.pr_auc_score(y_val, val_proba)

        return objective

    def _calculate_scale_pos_weight(self, y):
        N_pos = int((y == 1).sum())
        N_neg = int((y == 0).sum())
        if N_pos == 0:
            raise ValueError("y contains no positive class; scale_pos_weight is undefined.")

        self.scale_pos_weight = N_neg / N_pos
        return self.scale_pos_weight
    
    def build_model(self, name: str, params: Dict[str, Any], early_stopping_rounds: Optional[int] = None) -> Any:
        if early_stopping_rounds is None:
            early_stopping_rounds = self.early_stopping_rounds

        if name == "xgb":
            return XGBClassifier(
                **params,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="aucpr",
                early_stopping_rounds=early_stopping_rounds,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

        if name == "lgbm":
            return LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        if name == "cat":
            return CatBoostClassifier(
                **params,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="PRAUC",
                early_stopping_rounds=early_stopping_rounds,
                random_state=self.random_state,
                has_time=True,
                verbose=False,
                allow_writing_files=False,
            )

        raise ValueError(f"Unknown model: {name}")
    
    def suggest_params(self, trial: optuna.Trial, name: str) -> Dict[str, Any]:
        if name == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }

        if name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

        if name == "cat":
            return {
                "iterations": trial.suggest_int("iterations", 300, 2000, step=100),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            }

        raise ValueError(f"Unknown model: {name}")
    
    @staticmethod
    def _ensure_dataframe(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        return pd.DataFrame(X).reset_index(drop=True)

    @staticmethod
    def _ensure_array(y: Any) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.Index)):
            return y.to_numpy()
        return np.asarray(y)

    @staticmethod
    def pr_auc_score(y_true: np.ndarray, proba: np.ndarray) -> float:
        return average_precision_score(y_true, proba)

    @staticmethod
    def evaluate_binary(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        pred = (proba >= threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
            "f1": f1_score(y_true, pred, zero_division=0),
        }

    @staticmethod
    def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        if len(thresholds) == 0:
            return 0.5

        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best_idx = int(np.nanargmax(f1))
        return float(thresholds[best_idx])
    
    def _extract_best_iteration(self, name: str, model: Any) -> int:
        if name == "xgb":
            best_iteration = getattr(model, "best_iteration", None)
            return int(best_iteration + 1) if best_iteration is not None else int(model.get_params()["n_estimators"])

        if name == "lgbm":
            best_iteration = getattr(model, "best_iteration_", None)
            return int(best_iteration) if best_iteration is not None else int(model.get_params()["n_estimators"])

        if name == "cat":
            best_iteration = model.get_best_iteration()
            return int(best_iteration) if best_iteration is not None and best_iteration > 0 else int(model.get_params()["iterations"])

        raise ValueError(f"Unknown model: {name}")

    def _build_refit_model(self, name: str, params: Dict[str, Any], best_iteration: int) -> Any:
        refit_params = params.copy()

        if name == "xgb":
            refit_params["n_estimators"] = best_iteration
            return XGBClassifier(
                **refit_params,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="aucpr",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

        if name == "lgbm":
            refit_params["n_estimators"] = best_iteration
            return LGBMClassifier(
                **refit_params,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        if name == "cat":
            refit_params["iterations"] = best_iteration
            return CatBoostClassifier(
                **refit_params,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="PRAUC",
                random_state=self.random_state,
                has_time=True,
                verbose=False,
                allow_writing_files=False,
            )

        raise ValueError(f"Unknown model: {name}")

    def _make_oof_meta_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        tscv = TimeSeriesSplit(n_splits=self.meta_splits)
        oof = np.full((len(X), len(self.model_names)), np.nan, dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
            print(f"OOF fold {fold}/{self.meta_splits} | train={len(tr_idx):,} val={len(va_idx):,}")
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            for col_idx, name in enumerate(self.model_names):
                fold_model = self.build_model(
                    name=name,
                    params=self.best_params[name],
                    early_stopping_rounds=self.oof_early_stopping_rounds,
                )
                fold_model = self.fit_model(name, fold_model, X_tr, y_tr, X_va, y_va)
                oof[va_idx, col_idx] = fold_model.predict_proba(X_va)[:, 1]

        valid_mask = ~np.isnan(oof).any(axis=1)
        oof_df = pd.DataFrame(oof[valid_mask], columns=[f"{name}_prob" for name in self.model_names])
        y_oof = y[valid_mask]
        return oof_df, y_oof

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
    ) -> "AMLEnsemble":
        X_train = self._ensure_dataframe(X_train)
        y_train = self._ensure_array(y_train)
        X_val = self._ensure_dataframe(X_val)
        y_val = self._ensure_array(y_val)

        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_val_ = X_val
        self.y_val_ = y_val
        self.X_trainval_ = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        self.y_trainval_ = np.concatenate([y_train, y_val])

        self.tune_models(X_train, y_train, X_val, y_val)

        self.base_models_train = {}
        self.best_iterations = {}
        for name in self.model_names:
            model = self.build_model(name=name, params=self.best_params[name], early_stopping_rounds=self.early_stopping_rounds)
            model = self.fit_model(name, model, X_train, y_train, X_val, y_val)
            self.base_models_train[name] = model
            self.best_iterations[name] = self._extract_best_iteration(name, model)

        oof_meta_df, y_oof = self._make_oof_meta_features(X_train, y_train)

        self.meta_model = SkPipeline([
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.random_state,
                ),
            ),
        ])
        self.meta_model.fit(oof_meta_df, y_oof)

        val_meta_df = pd.DataFrame(
            {
                f"{name}_prob": self.base_models_train[name].predict_proba(X_val)[:, 1]
                for name in self.model_names
            }
        )
        val_stack_proba = self.meta_model.predict_proba(val_meta_df)[:, 1]
        self.best_stack_threshold = self.best_f1_threshold(y_val, val_stack_proba)

        self.base_models_trainval = {}
        for name in self.model_names:
            refit_model = self._build_refit_model(
                name=name,
                params=self.best_params[name],
                best_iteration=self.best_iterations[name],
            )
            refit_model.fit(self.X_trainval_, self.y_trainval_)
            self.base_models_trainval[name] = refit_model

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.meta_model is None or not self.base_models_trainval:
            raise RuntimeError("The ensemble is not fitted yet. Call fit(...) first.")

        X = self._ensure_dataframe(X)
        meta_df = pd.DataFrame(
            {
                f"{name}_prob": self.base_models_trainval[name].predict_proba(X)[:, 1]
                for name in self.model_names
            }
        )
        return self.meta_model.predict_proba(meta_df)[:, 1]

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        if threshold is None:
            threshold = self.best_stack_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: Any, y: Any, threshold: Optional[float] = None) -> Dict[str, float]:
        y = self._ensure_array(y)
        if threshold is None:
            threshold = self.best_stack_threshold
        proba = self.predict_proba(X)
        return self.evaluate_binary(y, proba, threshold=threshold)

    def evaluate_base_models(self, X: Any, y: Any) -> pd.DataFrame:
        if not self.base_models_trainval:
            raise RuntimeError("Base models are not fitted yet. Call fit(...) first.")

        X = self._ensure_dataframe(X)
        y = self._ensure_array(y)

        rows = []
        for name in self.model_names:
            proba = self.base_models_trainval[name].predict_proba(X)[:, 1]
            metrics = self.evaluate_binary(y, proba, threshold=0.5)
            rows.append(
                {
                    "model": f"{name}_tuned",
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1_05": metrics["f1"],
                }
            )

        stack_metrics = self.evaluate(X, y)
        rows.append(
            {
                "model": "stacking_lr_clean",
                "roc_auc": stack_metrics["roc_auc"],
                "pr_auc": stack_metrics["pr_auc"],
                "f1_05": stack_metrics["f1"],
            }
        )
        return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
