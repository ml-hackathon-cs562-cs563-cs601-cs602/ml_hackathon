"""Pipeline Stage 6: HMM & Viterbi route decoding and decoder tuning.

Owner: Member B
Stage: HMM and Viterbi Decoding
"""

from pathlib import Path
import json
import pickle
import time
import numpy as np
import optuna
import pandas as pd

from models.eval_harness import load_partition
from pathsnap.evaluation.metrics import observation_top1_accuracy, route_continuity
from pathsnap.hmm.emissions import prob_matrix
from pathsnap.hmm.viterbi import decode, DEFAULT_PARAMS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def seq_f1(true_route: np.ndarray, pred_route: np.ndarray) -> float:
    """Compute sequence-level arc set F1 score."""
    s = set(pred_route[pred_route >= 0].tolist())
    r = set(true_route.tolist())
    inter = len(s & r)
    pr = inter / max(len(s), 1)
    rc = inter / max(len(r), 1)
    return float(2 * pr * rc / max(pr + rc, 1e-9))


def evaluate_trip_hmm(meta: dict, probs: np.ndarray, params: dict) -> dict:
    """Evaluate classifier-only vs HMM on a single trip."""
    M = meta["M"]
    obs_ids = M[:, 0].astype(int)
    arc_ids = M[:, 1].astype(int)
    src_nodes = M[:, 2].astype(int)
    dst_nodes = M[:, 3].astype(int)
    truth_ids = meta["truth_ids"]
    arcs = meta["arcs"]
    track = meta["track"]
    route = meta["route"]

    # Candidate matrix format for HMM
    unique_obs = np.unique(obs_ids)
    K = 8
    cand_matrix = -np.ones((len(track), K), dtype=int)
    pm = np.full((len(track), K), 1e-4, dtype=np.float32)

    for i in range(len(track)):
        ix = np.flatnonzero(obs_ids == i)
        kk = min(len(ix), K)
        cand_matrix[i, :kk] = arc_ids[ix[:kk]]
        pm[i, :kk] = probs[ix[:kk]]

    # Classifier Top-1
    clf_pred = np.array([
        arc_ids[ix[np.argmax(probs[ix])]] if len(ix) > 0 else -1
        for ix in [np.flatnonzero(obs_ids == i) for i in range(len(track))]
    ], dtype=int)

    # HMM Viterbi
    hmm_pred = decode(arcs, cand_matrix, pm, track[:, 2], params)

    valid = truth_ids >= 0
    clf_top1 = float(np.mean(clf_pred[valid] == truth_ids[valid])) if valid.any() else 0.0
    hmm_top1 = float(np.mean(hmm_pred[valid] == truth_ids[valid])) if valid.any() else 0.0

    return {
        "trip_id": meta["trip_id"],
        "n_obs": len(track),
        "clf_top1": clf_top1,
        "clf_cont": route_continuity(src_nodes[clf_pred], dst_nodes[clf_pred]) if len(clf_pred) > 1 else 1.0,
        "clf_seqF1": seq_f1(route, clf_pred),
        "hmm_top1": hmm_top1,
        "hmm_cont": route_continuity(src_nodes[hmm_pred], dst_nodes[hmm_pred]) if len(hmm_pred) > 1 else 1.0,
        "hmm_seqF1": seq_f1(route, hmm_pred),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and student-test partition for HMM decoding...")
    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists():
        model_path = ROOT / "artifacts" / "randomforest.joblib"

    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        import joblib
        model = joblib.load(model_path)

    student_test = load_partition("student_test")

    # Load or default HMM parameters
    hmm_params_file = MODELS_DIR / "hmm_params.json"
    if hmm_params_file.exists():
        with open(hmm_params_file, "r") as f:
            params = json.load(f)
    else:
        params = DEFAULT_PARAMS

    rows = []
    for meta in student_test["metas"]:
        p = model.predict_proba(meta["X"])
        p_pos = p[:, 1] if p.ndim == 2 else p
        res = evaluate_trip_hmm(meta, p_pos, params)
        rows.append(res)

    df = pd.DataFrame(rows)
    w = df["n_obs"]
    comp = {
        "clf_top1": float(np.average(df["clf_top1"], weights=w)),
        "clf_cont": float(df["clf_cont"].mean()),
        "clf_seqF1": float(df["clf_seqF1"].mean()),
        "hmm_top1": float(np.average(df["hmm_top1"], weights=w)),
        "hmm_cont": float(df["hmm_cont"].mean()),
        "hmm_seqF1": float(df["hmm_seqF1"].mean()),
    }

    comp_df = pd.DataFrame([comp])
    comp_df.to_csv(RESULTS_DIR / "hmm_comparison.csv", index=False)
    print("\nClassifier vs HMM Results:")
    print(comp_df.round(4).to_string())
    print("Stage 6 HMM Viterbi decoding completed.")


if __name__ == "__main__":
    main()
