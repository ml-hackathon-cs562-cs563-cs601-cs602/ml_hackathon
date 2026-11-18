"""Train and evaluate Random Forest (Member C)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from models.eval_harness import (
    MAX_TRAIN_OBS_PER_TRIP,
    evaluate_experiment,
    load_partition,
)

CONFIG = {
    "algorithm_family": "tree ensemble (bagging)",
    "initial_hyperparameters": {
        "n_estimators": 20,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "max_samples": 0.5,
        "class_weight": "balanced_subsample",
        "bootstrap": True,
    },
    "preprocessing": ["SimpleImputer(strategy=median)"],
    "objective": "Gini impurity reduction at each split; class probabilities aggregated across trees",
    "random_state": 42,
}


def main():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=20,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=5,
                max_features="sqrt",
                max_samples=0.5,
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    train = load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    test = load_partition("student_test")
    evaluate_experiment(model, "RandomForest", CONFIG, train, test)


if __name__ == "__main__":
    main()
