from pathlib import Path
import pandas as pd

CANDIDATES = Path("data/interim/candidates_v0.parquet")
ALIGNMENTS = Path("results/alignment/train_v0/point_alignment_labels.csv")
OUTPUT = Path("data/features/candidate_labeled_train_v0.parquet")
METRICS = Path("results/alignment/train_v0/candidate_coverage_metrics.csv")

# Read source data
candidates = pd.read_parquet(CANDIDATES)
labels = pd.read_csv(ALIGNMENTS, dtype={"trip_id": str})

# Validate required columns
required_candidates = {
    "trip_id", "partition", "point_idx", "candidate_arc_id"
}
required_labels = {
    "trip_id", "point_idx", "aligned_arc_id",
    "alignment_status", "alignment_distance_m", "confidence"
}

assert required_candidates.issubset(candidates.columns)
assert required_labels.issubset(labels.columns)

# Training partition only
candidates = candidates[candidates["partition"] == "train"].copy()

# Standardize identifiers
candidates["trip_id"] = candidates["trip_id"].astype(str).str.zfill(8)
labels["trip_id"] = labels["trip_id"].astype(str).str.zfill(8)

candidates["point_idx"] = pd.to_numeric(
    candidates["point_idx"], errors="raise"
).astype("int64")
candidates["candidate_arc_id"] = pd.to_numeric(
    candidates["candidate_arc_id"], errors="raise"
).astype("int64")

labels["point_idx"] = pd.to_numeric(
    labels["point_idx"], errors="raise"
).astype("int64")
labels["aligned_arc_id"] = pd.to_numeric(
    labels["aligned_arc_id"], errors="coerce"
)

# Validate uniqueness
candidate_keys = [
    "trip_id", "partition", "point_idx", "candidate_arc_id"
]
assert not candidates.duplicated(candidate_keys).any(), \
    "Duplicate candidate rows detected"

label_keys = ["trip_id", "point_idx"]
assert not labels.duplicated(label_keys).any(), \
    "Duplicate alignment labels detected"

# Only strict aligned observations are eligible
eligible = labels[
    labels["alignment_status"] == "aligned"
].copy()

eligible = eligible.dropna(subset=["aligned_arc_id"])
eligible["aligned_arc_id"] = eligible["aligned_arc_id"].astype("int64")

# Determine whether each eligible point has its reference arc
candidate_check = candidates.merge(
    eligible[
        ["trip_id", "point_idx", "aligned_arc_id"]
    ],
    on=["trip_id", "point_idx"],
    how="inner",
    validate="many_to_one"
)

candidate_check["is_reference_arc"] = (
    candidate_check["candidate_arc_id"]
    == candidate_check["aligned_arc_id"]
)

point_hits = (
    candidate_check.groupby(["trip_id", "point_idx"])
    ["is_reference_arc"].any()
    .rename("candidate_hit")
    .reset_index()
)

point_results = eligible.merge(
    point_hits,
    on=["trip_id", "point_idx"],
    how="left",
    validate="one_to_one"
)

point_results["candidate_hit"] = (
    point_results["candidate_hit"].fillna(False).astype(bool)
)

# Calculate per-trip candidate coverage
per_trip = point_results.groupby("trip_id").agg(
    eligible_points=("point_idx", "count"),
    candidate_hits=("candidate_hit", "sum")
).reset_index()

per_trip["candidate_misses"] = (
    per_trip["eligible_points"] - per_trip["candidate_hits"]
)

per_trip["candidate_hit_rate_pct"] = (
    100 * per_trip["candidate_hits"] / per_trip["eligible_points"]
)

# Exclude known anomalous trip 00000054 from supervised training
trainable_points = point_results[
    (point_results["candidate_hit"])
    & (point_results["trip_id"] != "00000054")
].copy()

# Keep candidate rows only for trainable GPS points
trainable = candidates.merge(
    trainable_points[
        [
            "trip_id", "point_idx", "aligned_arc_id",
            "alignment_distance_m", "confidence"
        ]
    ],
    on=["trip_id", "point_idx"],
    how="inner",
    validate="many_to_one"
)

# Binary candidate classification label
trainable["label"] = (
    trainable["candidate_arc_id"]
    == trainable["aligned_arc_id"]
).astype("int8")

# Every retained GPS point must have exactly one positive candidate
positive_counts = trainable.groupby(
    ["trip_id", "point_idx"]
)["label"].sum()

assert (positive_counts == 1).all(), \
    "A retained GPS point does not have exactly one positive candidate"

assert set(trainable["partition"].unique()) == {"train"}
assert "00000054" not in set(trainable["trip_id"].unique())

# Save output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
trainable.to_parquet(OUTPUT, index=False)

# Save per-trip metrics
per_trip.to_csv(METRICS, index=False)

# Print summary
total_eligible = len(point_results)
total_hits = int(point_results["candidate_hit"].sum())
total_misses = total_eligible - total_hits

print("\n========== CANDIDATE COVERAGE ==========")
print("Eligible aligned GPS points:", total_eligible)
print("Candidate hits:", total_hits)
print("Candidate misses:", total_misses)
print("Candidate hit rate (%):", round(100 * total_hits / total_eligible, 2))
print("Candidate miss rate (%):", round(100 * total_misses / total_eligible, 2))

print("\n========== SUPERVISED TRAINING DATA ==========")
print("Training candidate rows:", len(trainable))
print("Unique GPS points:", trainable[["trip_id", "point_idx"]].drop_duplicates().shape[0])
print("Positive labels:", int(trainable["label"].sum()))
print("Negative labels:", int((trainable["label"] == 0).sum()))
print("Excluded anomalous trip: 00000054")

print("\n========== OUTPUTS ==========")
print(OUTPUT)
print(METRICS)
