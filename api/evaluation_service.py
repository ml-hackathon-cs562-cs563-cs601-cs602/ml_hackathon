"""Server-side evaluation service for PathSnap routes.

The ONLY module allowed to read ground-truth routes for evaluation.
Owner: Member A
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from pathsnap.evaluation.metrics import route_continuity
from pathsnap.ingest.parsers import load_trip
from pathsnap.labels.route_labels import align_labels

ROOT = Path(__file__).resolve().parents[1]


def score_prediction(root: str, tid: str, res: dict) -> Optional[Dict[str, Any]]:
    """Compute server-side evaluation metrics against reference route.
    
    Checks api/server_data/ first for official evaluator trips, then falls back to root.
    """
    server_data = ROOT / "api" / "server_data" / tid
    route_dir = server_data if server_data.exists() else Path(root)

    try:
        _, _, _, rt = load_trip(route_dir, tid, with_route=True)
    except Exception:
        rt = None

    if rt is None or len(rt) == 0:
        return None

    P = res["net"].proj(res["tr"][:, 0], res["tr"][:, 1])
    ref, _ = align_labels(res["net"], P, rt)
    ar = res["ar"]

    out = {}
    for k in ("clf", "pred"):
        pred_arcs = res[k]
        top1 = float(np.mean(pred_arcs == ref))
        src = ar[pred_arcs, 0] if len(pred_arcs) > 0 and pred_arcs.max() < len(ar) else []
        dst = ar[pred_arcs, 1] if len(pred_arcs) > 0 and pred_arcs.max() < len(ar) else []
        cont = route_continuity(src, dst) if len(src) > 1 else 1.0
        out[k] = {"top1_accuracy": top1, "route_continuity": cont}

    # Sequence F1
    s = set(res["pred"][res["pred"] >= 0].tolist())
    r = set(rt.tolist())
    inter = len(s & r)
    pr = inter / max(len(s), 1)
    rc = inter / max(len(r), 1)
    out["route_arc_precision"] = float(pr)
    out["route_arc_recall"] = float(rc)
    out["route_arc_f1"] = float(2 * pr * rc / max(pr + rc, 1e-9))
    return out
