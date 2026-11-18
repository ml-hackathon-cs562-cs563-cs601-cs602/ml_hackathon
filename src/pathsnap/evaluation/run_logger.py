"""Experiment Run Logger for PathSnap.

Owner: Member C
Phase 1: Shared harness

Implements standardized experiment logging to experiments/runs/ following the schema
defined in experiments/README.md:
- run_id
- model
- family
- author
- config_path
- seed
- features_version
- cv_macro_f1
- student_top1
- student_macro_f1
- log_loss
- train_time_s
- infer_time_s
- git_commit
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
import pandas as pd


def get_git_commit(cwd: Optional[Path] = None) -> str:
    """Return the current short git commit hash, or 'unknown' if not in git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def log_run(
    run_id: str,
    model: str,
    family: str,
    author: str,
    config_path: str,
    seed: int,
    features_version: str,
    cv_macro_f1: float,
    student_top1: float,
    student_macro_f1: float,
    log_loss: float,
    train_time_s: float,
    infer_time_s: float,
    git_commit: Optional[str] = None,
    output_dir: Optional[Path] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write an experiment run record JSON adhering to experiments/README.md.
    
    Args:
        run_id: Unique run identifier, e.g. 'run_logreg_v0'.
        model: Model name, e.g. 'LogisticRegression'.
        family: Algorithm family, e.g. 'linear', 'ensemble'.
        author: Author name or member id, e.g. 'Member C'.
        config_path: Path to model configuration file.
        seed: Random seed used for reproducibility.
        features_version: Features version string, e.g. 'v0'.
        cv_macro_f1: Validation candidate Macro F1 mean.
        student_top1: Student-test observation Top-1 accuracy.
        student_macro_f1: Student-test candidate/observation Macro F1.
        log_loss: Student-test candidate log loss.
        train_time_s: Training wall clock time in seconds.
        infer_time_s: Total inference wall clock time in seconds.
        git_commit: Git commit hash (defaults to current HEAD).
        output_dir: Directory where runs are stored (default: experiments/runs/).
        extra_metrics: Optional additional metric fields.
        
    Returns:
        Path to the written JSON file.
    """
    if output_dir is None:
        # Default to repo_root/experiments/runs
        repo_root = Path(__file__).resolve().parents[3]
        runs_dir = repo_root / "experiments" / "runs"
    else:
        runs_dir = Path(output_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)

    if git_commit is None:
        git_commit = get_git_commit(runs_dir)

    record: Dict[str, Any] = {
        "run_id": str(run_id),
        "model": str(model),
        "family": str(family),
        "author": str(author),
        "config_path": str(config_path),
        "seed": int(seed),
        "features_version": str(features_version),
        "cv_macro_f1": float(cv_macro_f1),
        "student_top1": float(student_top1),
        "student_macro_f1": float(student_macro_f1),
        "log_loss": float(log_loss),
        "train_time_s": float(train_time_s),
        "infer_time_s": float(infer_time_s),
        "git_commit": str(git_commit),
    }

    if extra_metrics:
        record["extra_metrics"] = extra_metrics

    file_path = runs_dir / f"{run_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return file_path


def load_run(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load an individual experiment run record."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_runs(runs_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all experiment run records from experiments/runs/ into a DataFrame."""
    if runs_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        runs_dir = repo_root / "experiments" / "runs"

    if not runs_dir.exists():
        return pd.DataFrame()

    records = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            records.append(load_run(p))
        except Exception:
            pass

    return pd.DataFrame(records)
