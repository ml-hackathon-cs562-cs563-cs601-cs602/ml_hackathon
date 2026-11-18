"""Train and evaluate Decision Tree (Member C)."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from models.eval_harness import (
    MAX_TRAIN_OBS_PER_TRIP,
    evaluate_experiment,
    load_partition,
)

CONFIG = {
    "algorithm_family": "tree-based",
    "initial_hyperparameters": {
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "preprocessing": ["SimpleImputer(strategy=median)"],
    "objective": "Gini impurity reduction at each split",
    "random_state": 42,
}


def main():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])
    train = load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    test = load_partition("student_test")
    evaluate_experiment(model, "DecisionTree", CONFIG, train, test)


if __name__ == "__main__":
    main()
