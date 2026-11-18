"""Model Factory for PathSnap.

Build a model object from a YAML config in configs/models/. One place that maps
config names to classes, so pipelines never import models directly.
Owner: Member C
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import yaml

from models.dtree.model import DecisionTreeModel
from models.logreg.model import LogisticRegressionModel
from models.rf.model import RandomForestModel


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_model(config_or_path: Union[str, Path, Dict[str, Any]]) -> Any:
    """Build a model object from a configuration dictionary or YAML file path.
    
    Supported models:
    - logreg / LogisticRegression -> LogisticRegressionModel
    - dtree / DecisionTree -> DecisionTreeModel
    - rf / RandomForest -> RandomForestModel
    - lgbm / LightGBM -> LightGBMModel (Member D)
    - mlp / MLP -> MLPModel (Member D)
    """
    if isinstance(config_or_path, (str, Path)):
        p = Path(config_or_path)
        config = _load_yaml(p)
        config_name = p.stem.lower()
    else:
        config = config_or_path
        config_name = str(config.get("model_name", config.get("model_type", ""))).lower()

    # Determine model key
    model_key = config.get("model_name", config.get("model_type", config_name)).lower()

    if "logreg" in model_key or "logistic" in model_key:
        return LogisticRegressionModel(config)
    elif "dtree" in model_key or "decision" in model_key or "tree" in model_key:
        return DecisionTreeModel(config)
    elif "rf" in model_key or "random" in model_key or "forest" in model_key:
        return RandomForestModel(config)
    elif "lgbm" in model_key or "lightgbm" in model_key:
        from models.lgbm.model import LightGBMModel
        return LightGBMModel(config)
    elif "mlp" in model_key:
        from models.mlp.model import MLPModel
        return MLPModel(config)
    else:
        raise ValueError(
            f"Unknown model identifier '{model_key}'. Supported: logreg, dtree, rf, lgbm, mlp"
        )
