"""Tests for Member C model wrappers and factory.

Owner: Member C
"""

from pathlib import Path
import numpy as np
import pytest

from models.dtree.model import DecisionTreeModel
from models.factory import get_model
from models.logreg.model import LogisticRegressionModel
from models.rf.model import RandomForestModel


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.choice([0, 1], size=100)
    return X, y


def test_logistic_regression_wrapper(dummy_data, tmp_path: Path):
    X, y = dummy_data
    model = LogisticRegressionModel()
    model.fit(X, y)

    assert model.train_time_s is not None
    assert model.train_time_s > 0

    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = model.predict(X, threshold=0.5)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Save and load
    save_path = tmp_path / "logreg.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = LogisticRegressionModel.load(save_path)
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs)


def test_decision_tree_wrapper(dummy_data, tmp_path: Path):
    X, y = dummy_data
    model = DecisionTreeModel()
    model.fit(X, y)

    assert model.train_time_s is not None
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    save_path = tmp_path / "dtree.joblib"
    model.save(save_path)
    loaded = DecisionTreeModel.load(save_path)
    np.testing.assert_allclose(probs, loaded.predict_proba(X))


def test_random_forest_wrapper(dummy_data, tmp_path: Path):
    X, y = dummy_data
    model = RandomForestModel()
    model.fit(X, y)

    assert model.train_time_s is not None
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    save_path = tmp_path / "rf.joblib"
    model.save(save_path)
    loaded = RandomForestModel.load(save_path)
    np.testing.assert_allclose(probs, loaded.predict_proba(X))


def test_model_factory():
    m_logreg = get_model({"model_name": "LogisticRegression"})
    assert isinstance(m_logreg, LogisticRegressionModel)

    m_dtree = get_model({"model_name": "DecisionTree"})
    assert isinstance(m_dtree, DecisionTreeModel)

    m_rf = get_model({"model_name": "RandomForest"})
    assert isinstance(m_rf, RandomForestModel)
