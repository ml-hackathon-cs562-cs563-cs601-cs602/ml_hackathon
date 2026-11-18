"""Smoke test: shared harness loads candidates and GroupKFold is trip-safe."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from models.eval_harness import load_partition


def main():
    data = load_partition("train", max_observations_per_trip=20)
    X, y, groups = data["X"], data["y"], data["groups"]
    assert X.ndim == 2 and X.shape[1] == len(data["feature_names"]), "Feature dimension mismatch"
    assert len(X) == len(y) == len(groups), "Row count mismatch"

    cv = GroupKFold(n_splits=5)
    scores = []
    for tr, va in cv.split(X, y, groups):
        assert not set(groups[tr]).intersection(set(groups[va])), "Trip leakage across folds!"
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X[tr], y[tr])
        scores.append(float(np.mean(dummy.predict(X[va]) == y[va])))

    print("PASS: candidate feature matrix loaded:", X.shape)
    print("PASS: 5 GroupKFold splits have no trip leakage")
    print("PASS: dummy model fitted and predicted in every fold")
    print("Dummy fold accuracies:", [round(x, 4) for x in scores])


if __name__ == "__main__":
    main()
