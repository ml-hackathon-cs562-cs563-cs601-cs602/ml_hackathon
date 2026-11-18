"""Pipeline Stage 7: Export final model and inference assets.

Owner: Member D
Stage: Export Artifacts
"""

from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting inference assets...")

    # Export final model and hmm params
    if (MODELS_DIR / "final_model.pkl").exists():
        shutil.copy2(MODELS_DIR / "final_model.pkl", ARTIFACTS_DIR / "final_model.pkl")
        print("Exported models/final_model.pkl -> artifacts/final_model.pkl")

    if (MODELS_DIR / "hmm_params.json").exists():
        shutil.copy2(MODELS_DIR / "hmm_params.json", ARTIFACTS_DIR / "hmm_params.json")
        print("Exported models/hmm_params.json -> artifacts/hmm_params.json")

    print("Stage 7 export completed.")


if __name__ == "__main__":
    main()
