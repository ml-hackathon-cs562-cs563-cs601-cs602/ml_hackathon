"""PathSnap Evaluation Module.

Owner: Member C
"""

from pathsnap.evaluation.metrics import (
    candidate_macro_f1,
    candidate_precision,
    candidate_recall,
    candidate_accuracy,
    candidate_log_loss,
    observation_top1_accuracy,
    candidate_miss_rate,
    candidate_coverage,
    observation_macro_f1,
    route_continuity,
    sequence_exact_match,
    compute_candidate_metrics,
    compute_observation_predictions,
    compute_route_metrics,
)
from pathsnap.evaluation.cv import (
    get_group_kfold,
    validate_no_leakage,
    split_by_trip,
    cross_validate,
)
from pathsnap.evaluation.run_logger import (
    log_run,
    load_run,
    load_all_runs,
    get_git_commit,
)

__all__ = [
    "candidate_macro_f1",
    "candidate_precision",
    "candidate_recall",
    "candidate_accuracy",
    "candidate_log_loss",
    "observation_top1_accuracy",
    "candidate_miss_rate",
    "candidate_coverage",
    "observation_macro_f1",
    "route_continuity",
    "sequence_exact_match",
    "compute_candidate_metrics",
    "compute_observation_predictions",
    "compute_route_metrics",
    "get_group_kfold",
    "validate_no_leakage",
    "split_by_trip",
    "cross_validate",
    "log_run",
    "load_run",
    "load_all_runs",
    "get_git_commit",
]
