"""Pipeline Stage 3: Feature extraction for candidate scoring.

Owner: Member B
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from pathsnap.features.build import trip_table, FEATS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "features"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building candidate feature tables...")

    for part in ["train", "student_test"]:
        part_dir = DATA_DIR / part
        if not part_dir.exists():
            continue
        tables = {}
        trips = sorted(p.name for p in part_dir.iterdir() if p.is_dir())
        for tid in trips:
            tab = trip_table(str(part_dir), tid, with_route=True)
            tables[tid] = tab

        pkl_path = OUT_DIR / f"{part}_features.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(tables, f)
        print(f"Saved {len(tables)} trips to {pkl_path}")

    print("Stage 3 feature extraction completed.")


if __name__ == "__main__":
    main()
