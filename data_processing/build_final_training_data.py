from pathlib import Path
import pandas as pd

KEYS = [
    "trip_id",
    "partition",
    "point_idx",
    "candidate_arc_id",
]

LABEL_PATH = Path("data/features/candidate_labeled_train_v0.parquet")
RAW_PATH = Path("data/features/features_v0.parquet")
SCALED_PATH = Path("data/features/features_v0_scaled.parquet")

RAW_OUTPUT = Path("data/features/candidate_labeled_features_train_v0.parquet")
SCALED_OUTPUT = Path("data/features/candidate_labeled_features_train_v0_scaled.parquet")

labels = pd.read_parquet(LABEL_PATH)
raw = pd.read_parquet(RAW_PATH)
scaled = pd.read_parquet(SCALED_PATH)

# Keep training partition only
raw = raw[raw["partition"] == "train"].copy()
scaled = scaled[scaled["partition"] == "train"].copy()

# Check candidate identity uniqueness
for name, df in [("labels", labels), ("raw", raw), ("scaled", scaled)]:
    duplicates = df.duplicated(KEYS).sum()
    print(f"{name} duplicate keys: {duplicates}")
    assert duplicates == 0, f"Duplicate keys in {name}"

# Select label information
label_columns = KEYS + [
    "aligned_arc_id",
    "alignment_distance_m",
    "confidence",
    "label",
]

label_data = labels[label_columns].copy()

# Join labels to raw features
raw_labeled = raw.merge(
    label_data,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

# Join labels to training-scaled features
scaled_labeled = scaled.merge(
    label_data,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

# Verify exact row alignment
assert len(raw_labeled) == len(labels)
assert len(scaled_labeled) == len(labels)

assert raw_labeled["label"].notna().all()
assert scaled_labeled["label"].notna().all()

assert set(raw_labeled["trip_id"].unique()) == set(labels["trip_id"].unique())
assert "00000054" not in set(raw_labeled["trip_id"].unique())

assert set(raw_labeled["partition"].unique()) == {"train"}
assert set(scaled_labeled["partition"].unique()) == {"train"}

# Verify one positive candidate per retained GPS point
positive_counts = raw_labeled.groupby(
    ["trip_id", "point_idx"]
)["label"].sum()

assert (positive_counts == 1).all()

# Save datasets
RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

raw_labeled.to_parquet(RAW_OUTPUT, index=False)
scaled_labeled.to_parquet(SCALED_OUTPUT, index=False)

print("\n========== FINAL DATASETS ==========")
print("Raw shape:", raw_labeled.shape)
print("Scaled shape:", scaled_labeled.shape)
print("Unique trips:", raw_labeled["trip_id"].nunique())
print("Unique GPS observations:",
      raw_labeled[["trip_id", "point_idx"]].drop_duplicates().shape[0])

print("\nLabel distribution:")
print(raw_labeled["label"].value_counts())

print("\nRaw feature columns:")
print(raw_labeled.columns.tolist())

print("\nScaled feature columns:")
print(scaled_labeled.columns.tolist())

print("\nSaved:")
print(RAW_OUTPUT)
print(SCALED_OUTPUT)
