"""Shared, leakage-safe evaluation harness for PathSnap candidate classifiers.

Owner: Member C
Phase 1: Shared harness & Phase 2: Baseline Models

Rows are road-edge candidates; `groups` identifies the originating trip. The
observation-level prediction is the candidate with the largest P(edge is on
reference route) within each GPS observation's candidate set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation.metrics import (
    candidate_accuracy,
    candidate_log_loss,
    candidate_macro_f1,
    candidate_precision,
    candidate_recall,
    compute_candidate_metrics,
    compute_observation_predictions,
    compute_route_metrics,
    observation_macro_f1,
    observation_top1_accuracy,
    route_continuity,
)
from pathsnap.evaluation.cv import cross_validate, split_by_trip
from pathsnap.evaluation.run_logger import log_run, get_git_commit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RUNS_DIR = ROOT / "experiments" / "runs"
K = 8
N_SPLITS = 5
MAX_TRAIN_OBS_PER_TRIP = 250
SEED = 42

FEATURES = [
    "dist_m",
    "heading_diff",
    "edge_bearing_sin",
    "edge_bearing_cos",
    "gps_speed",
    "gps_heading_sin",
    "gps_heading_cos",
    "edge_length",
    "src_degree",
    "dst_degree",
    "time_gap",
    "candidate_rank",
]


def _find_data_root() -> Path:
    """Locate dataset partition directory."""
    candidates = [
        ROOT / "data" / "raw",
        ROOT / "data" / "Student folder",
        ROOT / "data",
    ]
    for c in candidates:
        if (c / "train").exists():
            return c
    return ROOT / "data" / "raw"


def read_trip(folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read .track, .nodes, .arcs, and .route files for a trip."""
    tid = folder.name
    track_path = folder / f"{tid}.track"
    nodes_path = folder / f"{tid}.nodes"
    arcs_path = folder / f"{tid}.arcs"
    route_path = folder / f"{tid}.route"

    track = np.loadtxt(track_path)
    nodes = np.loadtxt(nodes_path)
    arcs = np.loadtxt(arcs_path, dtype=int)
    route = np.loadtxt(route_path, dtype=int).reshape(-1) if route_path.exists() else np.array([], dtype=int)

    if track.ndim == 1:
        track = track[None, :]
    if nodes.ndim == 1:
        nodes = nodes[None, :]
    if arcs.ndim == 1:
        arcs = arcs[None, :]

    return track, nodes, arcs, route


def make_trip_rows(
    folder: Path, k: int = K
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate candidate road-edge rows and features for a single trip."""
    track, nodes, arcs, route = read_trip(folder)

    # Local metric projection
    lon0, lat0 = np.nanmean(track[:, :2], axis=0)
    sx = 111320.0 * np.cos(np.deg2rad(lat0))
    sy = 110540.0
    xy = np.column_stack(((nodes[:, 0] - lon0) * sx, (nodes[:, 1] - lat0) * sy))
    gps = np.column_stack(((track[:, 0] - lon0) * sx, (track[:, 1] - lat0) * sy))

    src, dst = arcs[:, 0], arcs[:, 1]
    valid = (src >= 0) & (dst >= 0) & (src < len(nodes)) & (dst < len(nodes))
    arc_ids = np.flatnonzero(valid)
    src = src[valid]
    dst = dst[valid]

    a = xy[src]
    b = xy[dst]
    vec = b - a
    lens = np.linalg.norm(vec, axis=1)
    lens = np.maximum(lens, 1e-3)
    bearing = np.arctan2(vec[:, 1], vec[:, 0])
    mids = (a + b) / 2.0
    tree = cKDTree(mids)

    # Reference route edge indices
    rids = route[(route >= 0) & (route < len(arcs))]
    rids = rids[np.isin(rids, arc_ids)]
    if len(rids) > 0:
        ri = np.searchsorted(arc_ids, rids)
        ra = a[ri]
        rb = b[ri]
        rv = rb - ra
    else:
        ri = np.array([], dtype=int)

    # GPS dynamics
    delta = np.diff(gps, axis=0, prepend=gps[[0]])
    gps_heading = np.arctan2(delta[:, 1], delta[:, 0])
    dt = np.diff(track[:, 2], prepend=track[0, 2])
    dt = np.where(dt <= 0, 1.0, dt)
    speed = np.linalg.norm(delta, axis=1) / dt

    # Nearest K candidate road edges
    kk = min(k, len(mids))
    _, cand = tree.query(gps, k=kk)
    cand = np.asarray(cand).reshape(len(gps), kk)

    ca = a[cand]
    cb = b[cand]
    cv = cb - ca
    gp = gps[:, None, :]
    den = np.maximum((cv * cv).sum(axis=2), 1e-6)
    frac = np.clip(((gp - ca) * cv).sum(axis=2) / den, 0.0, 1.0)
    proj = ca + frac[:, :, None] * cv
    dist = np.linalg.norm(gp - proj, axis=2)
    cbear = bearing[cand]

    # Nearest point on supplied reference route
    if len(ri) > 0:
        rtree = cKDTree(mids[ri])
        kr = min(24, len(ri))
        _, rnear = rtree.query(gps, k=kr)
        rnear = np.asarray(rnear).reshape(len(gps), kr)
        rra = ra[rnear]
        rrv = rv[rnear]
        rden2 = np.maximum((rrv * rrv).sum(axis=2), 1e-6)
        rt = np.clip(((gp - rra) * rrv).sum(axis=2) / rden2, 0.0, 1.0)
        rproj = rra + rt[:, :, None] * rrv
        rd = np.linalg.norm(gp - rproj, axis=2)
        best = np.argmin(rd, axis=1)
        truth_ids = rids[rnear[np.arange(len(gps)), best]]
    else:
        truth_ids = np.full(len(gps), -1, dtype=int)

    hd = np.abs((gps_heading[:, None] - cbear + np.pi) % (2 * np.pi) - np.pi)
    rank = np.broadcast_to(np.arange(kk), (len(gps), kk))

    src_deg = np.bincount(src, minlength=len(nodes))
    dst_deg = np.bincount(dst, minlength=len(nodes))

    X = np.stack(
        [
            dist,
            hd,
            np.sin(cbear),
            np.cos(cbear),
            np.broadcast_to(speed[:, None], dist.shape),
            np.broadcast_to(np.sin(gps_heading)[:, None], dist.shape),
            np.broadcast_to(np.cos(gps_heading)[:, None], dist.shape),
            np.broadcast_to(lens[cand], dist.shape),
            np.broadcast_to(src_deg[src[cand]], dist.shape),
            np.broadcast_to(dst_deg[dst[cand]], dist.shape),
            np.broadcast_to(dt[:, None], dist.shape),
            rank,
        ],
        axis=2,
    ).reshape(-1, 12).astype(np.float32)

    obs = np.repeat(np.arange(len(gps)), kk)
    edge = arc_ids[cand].reshape(-1)
    ss = src[cand].reshape(-1)
    dd = dst[cand].reshape(-1)
    M = np.column_stack([obs, edge, ss, dd, dist.reshape(-1)]).astype(object)
    y = (edge == np.repeat(truth_ids, kk)).astype(np.int8)

    return X, y, M, track, nodes, arcs, route, truth_ids


def load_partition(
    partition: str,
    max_observations_per_trip: Optional[int] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Load candidate rows while retaining complete candidate groups per GPS fix."""
    data_root = _find_data_root()
    part_dir = data_root / partition
    if not part_dir.exists():
        raise FileNotFoundError(f"Dataset partition not found: {part_dir}")

    Xs, ys, groups, metas = [], [], [], []
    rng = np.random.default_rng(seed)

    for folder in sorted(p for p in part_dir.iterdir() if p.is_dir()):
        X, y, M, track, nodes, arcs, route, truth_ids = make_trip_rows(folder)
        obs_ids = M[:, 0].astype(int)

        if max_observations_per_trip is not None and len(track) > max_observations_per_trip:
            chosen_obs = np.sort(rng.choice(len(track), max_observations_per_trip, replace=False))
            keep = np.isin(obs_ids, chosen_obs)
            X, y, M = X[keep], y[keep], M[keep]

        trip_id = folder.name
        Xs.append(np.asarray(X, dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.int8))
        groups.extend([trip_id] * len(y))
        metas.append({
            "trip_id": trip_id,
            "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int8),
            "M": np.asarray(M),
            "track": track,
            "nodes": nodes,
            "arcs": arcs,
            "route": route,
            "truth_ids": truth_ids,
        })

    if not Xs:
        raise ValueError(f"No trips found in {part_dir}")

    return {
        "X": np.vstack(Xs),
        "y": np.concatenate(ys),
        "groups": np.asarray(groups),
        "metas": metas,
        "feature_names": FEATURES,
    }


def _positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    """Extract positive class probabilities reliably."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                return np.asarray(p[:, classes.index(1)], dtype=float)
            return np.zeros(len(X), dtype=float)
        elif p.ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=float)
        return np.asarray(p.ravel(), dtype=float)
    else:
        return np.asarray(model.predict(X), dtype=float)


def _predict_partition(model: Any, data: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float]]:
    """Predict probabilities and time inference per trip."""
    chunks, infer_times = [], []
    for meta in data["metas"]:
        start = time.perf_counter()
        p = _positive_probability(model, meta["X"])
        infer_times.append(time.perf_counter() - start)
        chunks.append(p)
    return chunks, infer_times


def evaluate_experiment(
    model: Any,
    model_name: str,
    config: Dict[str, Any],
    train: Optional[Dict[str, Any]] = None,
    student_test: Optional[Dict[str, Any]] = None,
    n_splits: int = N_SPLITS,
    save_model: bool = True,
) -> Dict[str, Any]:
    """Run GroupKFold CV, fit on full training set, and evaluate on untouched student test."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cm_dir = RESULTS / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    train = train or load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    student_test = student_test or load_partition("student_test", None)

    X, y, groups = train["X"], train["y"], train["groups"]
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))

    # 1. 5-fold GroupKFold CV
    cv_res = cross_validate(
        model, X, y, groups, n_splits=folds, predict_proba_fn=_positive_probability
    )
    cv_mean = cv_res["cv_macro_f1_mean"]
    cv_std = cv_res["cv_macro_f1_std"]
    cv_train_mean = cv_res["cv_train_macro_f1_mean"]
    fold_rows = cv_res["fold_rows"]

    # 2. Final fit on all training data
    fit_start = time.perf_counter()
    fitted = clone(model) if hasattr(model, "get_params") else model
    fitted.fit(X, y)
    training_seconds = time.perf_counter() - fit_start

    # Training metrics
    train_p = _positive_probability(fitted, X)
    train_cand = compute_candidate_metrics(y, train_p)

    # 3. Untouched student-test evaluation
    test_p_chunks, infer_times = _predict_partition(fitted, student_test)
    test_y = student_test["y"]
    test_p = np.concatenate(test_p_chunks)
    test_cand = compute_candidate_metrics(test_y, test_p)
    route = compute_route_metrics(student_test["metas"], test_p_chunks)

    # 4. Confusion matrix plot
    cm = confusion_matrix(test_y, (test_p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        cm, display_labels=["Not reference edge", "Reference edge"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} — Student-test candidate confusion matrix")
    fig.tight_layout()
    cm_path = cm_dir / f"{model_name.lower()}.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    row = {
        "model": model_name,
        "algorithm_family": config.get("algorithm_family", "N/A"),
        "configuration_json": json.dumps(config, sort_keys=True),
        "cv_macro_f1_mean": float(cv_mean),
        "cv_macro_f1_std": float(cv_std),
        "cv_train_macro_f1_mean": float(cv_train_mean),
        **test_cand,
        "student_test_macro_f1": test_cand["candidate_macro_f1"],
        "student_test_top1_accuracy": route["student_test_top1_accuracy"],
        "student_test_observation_macro_f1": route["student_test_observation_macro_f1"],
        "route_continuity": route["route_continuity"],
        "candidate_coverage": route["candidate_coverage"],
        "candidate_miss_rate": route["candidate_miss_rate"],
        "training_seconds": float(training_seconds),
        "inference_seconds_total": float(sum(infer_times)),
        "inference_seconds_per_candidate": float(sum(infer_times) / max(1, len(test_y))),
        "train_candidate_macro_f1": train_cand["candidate_macro_f1"],
        "overfit_gap_train_minus_cv": float(train_cand["candidate_macro_f1"] - cv_mean),
        "overfit_gap_cv_minus_student": float(cv_mean - test_cand["candidate_macro_f1"]),
        "cv_folds": folds,
        "training_trip_count": int(len(unique_groups)),
        "student_test_trip_count": int(len(student_test["metas"])),
        "training_candidate_rows": int(len(y)),
        "student_test_candidate_rows": int(len(test_y)),
        "confusion_matrix_path": str(cm_path.relative_to(ROOT)),
    }

    # Save to experiments.csv
    csv_path = RESULTS / "experiments.csv"
    frame = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if not frame.empty and "model" in frame.columns:
        frame = frame[frame["model"] != model_name]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(csv_path, index=False)

    # Save fold CSV
    pd.DataFrame(fold_rows).to_csv(RESULTS / f"{model_name.lower()}_cv_folds.csv", index=False)

    # Save standardized experiment run JSON to experiments/runs/
    run_id = f"run_{model_name.lower()}_v0"
    log_run(
        run_id=run_id,
        model=model_name,
        family=config.get("algorithm_family", "N/A"),
        author="Member C",
        config_path=f"configs/models/{model_name.lower()}.yaml",
        seed=SEED,
        features_version="v0",
        cv_macro_f1=float(cv_mean),
        student_top1=float(route["student_test_top1_accuracy"]),
        student_macro_f1=float(test_cand["candidate_macro_f1"]),
        log_loss=float(test_cand["candidate_log_loss"]),
        train_time_s=float(training_seconds),
        infer_time_s=float(sum(infer_times)),
        output_dir=RUNS_DIR,
        extra_metrics={
            "cv_macro_f1_std": float(cv_std),
            "route_continuity": float(route["route_continuity"]),
            "candidate_miss_rate": float(route["candidate_miss_rate"]),
        },
    )

    # Save model artifact and config
    if save_model:
        joblib.dump(fitted, ARTIFACTS / f"{model_name.lower()}.joblib")
        with open(ARTIFACTS / f"{model_name.lower()}_config.json", "w") as f:
            json.dump(config, f, indent=2)

    print(f"\nCompleted {model_name} evaluation successfully.")
    return row
