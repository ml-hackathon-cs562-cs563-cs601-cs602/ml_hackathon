"""Train and evaluate Gaussian Naive Bayes (Member C)."""

from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from models.eval_harness import (
    MAX_TRAIN_OBS_PER_TRIP,
    evaluate_experiment,
    load_partition,
)

CONFIG = {
    "algorithm_family": "probabilistic generative classifier",
    "initial_hyperparameters": {"var_smoothing": 1e-9},
    "preprocessing": ["SimpleImputer(strategy=median)"],
    "objective": "maximum likelihood estimation of class priors and per-feature Gaussian parameters",
    "assumption": "features are conditionally independent given the class",
    "probability_caveat": "Naive Bayes probabilities can be poorly calibrated when feature independence is violated; Log Loss is measured rather than assumed good.",
}


def main():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GaussianNB(var_smoothing=1e-9)),
    ])
    train = load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    test = load_partition("student_test")
    evaluate_experiment(model, "GaussianNB", CONFIG, train, test)


if __name__ == "__main__":
    main()
