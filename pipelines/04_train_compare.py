"""Pipeline Stage 4: Train and compare models with the shared harness.

Owners: Member C and Member D
Stage: Model Training and Cross-Validation
"""

import argparse
from pathlib import Path
import pandas as pd

from models.factory import get_model
from models.eval_harness import evaluate_experiment, load_partition, MAX_TRAIN_OBS_PER_TRIP
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "models"
RESULTS_DIR = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description="PathSnap Model Comparison Pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "dtree"],
        help="List of model keys from configs/models to evaluate",
    )
    args = parser.parse_args()

    print(f"Loading training and student-test data partitions...")
    train = load_partition("train", MAX_TRAIN_OBS_PER_TRIP)
    test = load_partition("student_test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_key in args.models:
        config_file = CONFIGS_DIR / f"{model_key}.yaml"
        if not config_file.exists():
            print(f"Skipping {model_key}: {config_file} not found.")
            continue

        print(f"\n--- Evaluating Model: {model_key} ---")
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        try:
            model = get_model(cfg)
            model_name = cfg.get("model_name", model_key)
            row = evaluate_experiment(model, model_name, cfg, train=train, student_test=test)
            results.append(row)
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")

    if results:
        comp_df = pd.DataFrame(results)
        comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        print(f"\nSaved comparison of {len(results)} models to results/model_comparison.csv")


if __name__ == "__main__":
    main()
