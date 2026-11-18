"""Train and evaluate Logistic Regression (Member C)."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.eval_harness import (
    MAX_TRAIN_OBS_PER_TRIP,
    evaluate_experiment,
    load_partition,
)

CONFIG = {
    "algorithm_family": "linear probabilistic classifier",
    "initial_hyperparameters": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 500,
        "class_weight": "balanced",
    },
    "preprocessing": ["SimpleImputer(strategy=median)", "StandardScaler"],
    "objective": "binary logistic negative log-likelihood (cross-entropy), optimized by lbfgs",
    "random_state": 42,
}


def main():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])
    train = load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    test = load_partition("student_test")
    evaluate_experiment(model, "LogisticRegression", CONFIG, train, test)


if __name__ == "__main__":
    main()
