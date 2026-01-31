from __future__ import annotations

import optuna
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

class AMLEnsemble:
    def __init__(self,
                 X_train, y_train, 
                 X_val, y_val
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def ceate_models(self):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )

        study.optimize(
            self._make_objective(name),
            n_trials=20,
            show_progress_bar=False,
        )

    def _make_objective(self, name):
        def objective(trial):
            params = self._suggest_params_optuna(trial, name)
            model = self._build_models(name, params, self.scale_pos_weight)

            fit_kwargs = {
                "X": self.X_train,
                "y": self.y_train,
                "eval_set": [(self.X_val, self.y_val)],
            }

            if name == "xg":
                fit_kwargs["verbose"] = False

            model.fit(**fit_kwargs)

            y_val_pred = model.predict_proba(self.X_val)[:, 1]
            score = average_precision_score(self.y_val, y_val_pred)

            if not (0.0 < score < 1.0):
                return 0.0

            return score

        return objective

    def _scale_pos_weight(self, y):
        N_pos = (y == 1).sum()
        N_neg = (y == 0).sum()

        self.scale_pos_weight = N_neg / N_pos
        return self.scale_pos_weight
    
    def _build_models(self, name, params):
        if name == "cat":
            return CatBoostClassifier(
                eval_metric="PRAUC",
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                iterations=3000,
                early_stopping_rounds=200,
                verbose=False,
                **params,
            )

        if name == "xg":
            return XGBClassifier(
                eval_metric="aucpr",
                scale_pos_weight=self.scale_pos_weight,
                n_jobs=-1,
                early_stopping_rounds=300,
                random_state=42,
                use_label_encoder=False,
                verbosity=0,
                **params,
            )

        if name == "ligh":
            return LGBMClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
                **params,
            )

        raise ValueError(f"Unknown model name: {name}")
    
    def _suggest_params_optuna(self, trial, name):
        if name == "cat":
            params = {
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 30, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            }

        elif name == "xg":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
            }

        elif name == "ligh":
            params = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "num_leaves": trial.suggest_int("num_leaves", 16, 96),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            }
        else:
            raise ValueError(f"Unknown model name: {name}")

        return params
    