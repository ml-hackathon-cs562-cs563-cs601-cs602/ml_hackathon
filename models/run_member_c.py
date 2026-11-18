"""Run Member C's complete experimental pipeline and regenerate artifacts.

Owner: Member C
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

commands = [
    [sys.executable, "-m", "models.test_eval_harness"],
    [sys.executable, "-m", "models.train_logistic"],
    [sys.executable, "-m", "models.train_random_forest"],
    [sys.executable, "-m", "models.train_decision_tree"],
    [sys.executable, "-m", "models.train_gaussian_nb"],
    [sys.executable, "-m", "models.build_member_c_report"],
]

for cmd in commands:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

print("\nMember C training, evaluation, and report generation completed successfully.")
