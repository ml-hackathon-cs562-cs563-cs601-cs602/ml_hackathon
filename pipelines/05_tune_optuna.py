"""Pipeline Stage 5: Bayesian hyperparameter optimisation (Optuna TPE).

Owner: Member D
Stage: Hyperparameter Tuning
"""

from pathlib import Path
import json
import pickle
import sys
import time
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from models.eval_harness import load_partition
from pathsnap.evaluation.metrics import candidate_macro_f1, compute_candidate_metrics, compute_route_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEED = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("Loading training data for Bayesian optimisation...")
    train = load_partition("train", max_observations_per_trip=250)
    student_test = load_partition("student_test")

    X, y, groups = train["X"], train["y"], train["groups"]
    folds = list(GroupKFold(4).split(X, y, groups))

    def make_model(params):
        return xgb.XGBClassifier(random_state=SEED, n_jobs=2, tree_method="hist", verbosity=0, **params)

    def cv_score(params):
        scores = []
        for tr, va in folds:
            m = make_model(params)
            m.fit(X[tr], y[tr])
            p_va = m.predict_proba(X[va])[:, 1]
            scores.append(f1_score(y[va], (p_va >= 0.5).astype(int), average="macro", zero_division=0))
        return float(np.mean(scores))

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        return cv_score(p)

    print(f"Running Optuna TPE search ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print("Best params:", best)
    print("Best CV Macro F1:", study.best_value)

    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
    with open(RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # Refit and evaluate tuned vs untuned
    untuned_params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6}
    rows = []
    sx, sy = student_test["X"], student_test["y"]

    for name, p in [("untuned", untuned_params), ("tuned", best)]:
        m = make_model(p)
        t_fit = time.perf_counter()
        m.fit(X, y)
        train_s = time.perf_counter() - t_fit

        t_inf = time.perf_counter()
        ps = m.predict_proba(sx)[:, 1]
        infer_s = time.perf_counter() - t_inf

        cand_m = compute_candidate_metrics(sy, ps)
        rows.append({
            "model": name,
            "params": json.dumps(p),
            "cv_macro_f1": cv_score(p),
            "test_macro_f1": cand_m["candidate_macro_f1"],
            "test_log_loss": cand_m["candidate_log_loss"],
            "train_time_s": train_s,
            "inference_time_s": infer_s,
        })
        with open(MODELS_DIR / f"xgb_{name}.pkl", "wb") as f:
            pickle.dump(m, f)

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    print("Stage 5 Bayesian optimisation completed successfully.")


if __name__ == "__main__":
    main()
