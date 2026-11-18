"""Tests for evaluation/run_logger.py schema conformance.

Owner: Member C
"""

import json
from pathlib import Path
import pytest

from pathsnap.evaluation.run_logger import (
    get_git_commit,
    load_all_runs,
    load_run,
    log_run,
)


def test_log_run_schema(tmp_path: Path):
    """Verify that log_run writes a JSON file with all required fields from experiments/README.md."""
    required_keys = {
        "run_id",
        "model",
        "family",
        "author",
        "config_path",
        "seed",
        "features_version",
        "cv_macro_f1",
        "student_top1",
        "student_macro_f1",
        "log_loss",
        "train_time_s",
        "infer_time_s",
        "git_commit",
    }

    file_path = log_run(
        run_id="test_run_01",
        model="LogisticRegression",
        family="linear",
        author="Member C",
        config_path="configs/models/logreg.yaml",
        seed=42,
        features_version="v0",
        cv_macro_f1=0.75,
        student_top1=0.82,
        student_macro_f1=0.78,
        log_loss=0.35,
        train_time_s=1.23,
        infer_time_s=0.45,
        git_commit="abcdef1",
        output_dir=tmp_path,
    )

    assert file_path.exists()
    record = load_run(file_path)

    # Check all required keys exist
    for k in required_keys:
        assert k in record, f"Missing required key: {k}"

    assert record["run_id"] == "test_run_01"
    assert record["model"] == "LogisticRegression"
    assert record["seed"] == 42
    assert record["cv_macro_f1"] == 0.75

    # Check load_all_runs
    df = load_all_runs(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "test_run_01"
