"""Pipeline Stage 2: Candidate road arc generation.

Owner: Member B
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pathsnap.candidates.generate import candidates, candidate_miss_rate, K_CAND, RADIUS
from pathsnap.geo.projection import Proj
from pathsnap.ingest.data_loader import Network
from pathsnap.ingest.parsers import load_trip

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"


def main():
    print(f"Generating candidate road edges (K={K_CAND}, radius={RADIUS}m)...")
    summary = []
    for part in ["train", "student_test"]:
        part_dir = DATA_DIR / part
        if not part_dir.exists():
            continue
        miss_rates = []
        for folder in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            tid = folder.name
            tr, nd, ar, rt = load_trip(part_dir, tid, with_route=True)
            proj = Proj(tr[:, 0].mean(), tr[:, 1].mean())
            net = Network(nd, ar, proj)
            P = proj(tr[:, 0], tr[:, 1])
            cand, dist, tpos = candidates(net, P, k=K_CAND, radius=RADIUS)

            if rt is not None and len(rt) > 0:
                mr = candidate_miss_rate(cand, rt[:len(cand)] if len(rt) >= len(cand) else np.resize(rt, len(cand)))
                miss_rates.append(mr)

        avg_mr = float(np.mean(miss_rates)) if miss_rates else 0.0
        summary.append({"partition": part, "avg_miss_rate": avg_mr})
        print(f"Partition {part}: Average candidate miss rate = {avg_mr:.4f}")

    print("Stage 2 candidate generation completed.")


if __name__ == "__main__":
    main()
