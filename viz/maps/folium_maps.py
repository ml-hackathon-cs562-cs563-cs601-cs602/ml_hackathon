"""Interactive Folium map generation for PathSnap routes.

Owner: Member A
"""

from pathlib import Path
import json
import pickle
import sys
import folium
import numpy as np

from pathsnap.candidates.generate import candidates
from pathsnap.features.build import build_features
from pathsnap.geo.projection import Proj
from pathsnap.hmm.emissions import prob_matrix
from pathsnap.hmm.viterbi import decode
from pathsnap.ingest.data_loader import Network
from pathsnap.ingest.parsers import load_trip

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MAPS_DIR = ROOT / "maps"


def generate_map(data_root: str, partition: str, trip_id: str, output_dir: Path = MAPS_DIR) -> Path:
    """Generate interactive Folium map for a single trip."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model & HMM params
    model_file = MODELS_DIR / "final_model.pkl"
    if not model_file.exists():
        model_file = ROOT / "artifacts" / "randomforest.joblib"

    if model_file.suffix == ".pkl":
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    else:
        import joblib
        model = joblib.load(model_file)

    hmm_file = MODELS_DIR / "hmm_params.json"
    with open(hmm_file, "r") as f:
        params = json.load(f)

    tr, nd, ar, rt = load_trip(Path(data_root) / partition, trip_id, with_route=True)
    proj = Proj(tr[:, 0].mean(), tr[:, 1].mean())
    net = Network(nd, ar, proj)
    P = proj(tr[:, 0], tr[:, 1])
    ts = tr[:, 2]

    cand, dist, tpos = candidates(net, P)
    X, ri, rk, a = build_features(net, P, ts, cand, dist, tpos)
    p = model.predict_proba(X)[:, 1]

    tab = {"P": P, "cand": cand, "ri": ri, "rk": rk, "arc": a}
    pm = prob_matrix(tab, p)
    pred_hmm = decode(ar, cand, pm, ts, params)

    # Collapse repeated consecutive edges
    collapse = lambda s: s[np.r_[True, s[1:] != s[:-1]]]
    ll = lambda arcs: [
        [[float(nd[ar[x, 0], 1]), float(nd[ar[x, 0], 0])], [float(nd[ar[x, 1], 1]), float(nd[ar[x, 1], 0])]]
        for x in arcs if 0 <= x < len(ar)
    ]

    m = folium.Map(location=[float(tr[:, 1].mean()), float(tr[:, 0].mean())], zoom_start=14, tiles="OpenStreetMap")

    # Road network near trajectory
    tree = net.tree()
    near = np.unique(net._ids[np.unique(np.concatenate(tree.query_ball_point(P[::5], 150)))])
    fg_net = folium.FeatureGroup("Road network (near trip)", show=True)
    for s in ll(near):
        folium.PolyLine(s, color="#888888", weight=1.5, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    # GPS observations
    fg_gps = folium.FeatureGroup("GPS observations", show=True)
    for lo, la in tr[::2, :2]:
        folium.CircleMarker([float(la), float(lo)], radius=2, color="#1f77ff", fill=True).add_to(fg_gps)
    fg_gps.add_to(m)

    # Reference route
    if rt is not None and len(rt) > 0:
        fg_ref = folium.FeatureGroup("Reference route (ground truth)", show=True)
        for s in ll(rt):
            folium.PolyLine(s, color="#2ca02c", weight=6, opacity=0.7).add_to(fg_ref)
        fg_ref.add_to(m)

    # HMM route
    valid_hmm = collapse(pred_hmm[pred_hmm >= 0])
    fg_hmm = folium.FeatureGroup("Classifier + HMM route", show=True)
    for s in ll(valid_hmm):
        folium.PolyLine(s, color="#e41a1c", weight=3.5).add_to(fg_hmm)
    fg_hmm.add_to(m)

    folium.LayerControl().add_to(m)

    legend_html = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:10px;border:1px solid #888;font:12px sans-serif;border-radius:4px">'
        '<b>PathSnap Route Legend</b><br>'
        '<span style="color:#1f77ff">&#9679;</span> GPS observations<br>'
        '<span style="color:#2ca02c">&#9644;&#9644;</span> Reference route<br>'
        '<span style="color:#e41a1c">&#9644;&#9644;</span> Reconstructed route (HMM)<br>'
        '<span style="color:#888888">&#9644;</span> Road network</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    m.fit_bounds([[float(tr[:, 1].min()), float(tr[:, 0].min())], [float(tr[:, 1].max()), float(tr[:, 0].max())]])

    out_file = output_dir / f"trip_{partition}_{trip_id}.html"
    m.save(str(out_file))
    print(f"Generated Folium map: {out_file}")
    return out_file


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw")
    part = sys.argv[2] if len(sys.argv) > 2 else "student_test"
    trip = sys.argv[3] if len(sys.argv) > 3 else "00000007"
    generate_map(data_dir, part, trip)
