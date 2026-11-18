"""Build comparison plots and Member C report sections.

Owner: Member C
Phase 2: Reporting (Sections 3.1 & 3.3)
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PLOTS = RES / "comparison_plots"
REP = ROOT / "report"
REP_SECTIONS = REP / "sections"


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    REP_SECTIONS.mkdir(parents=True, exist_ok=True)

    csv_path = RES / "experiments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run model training first.")

    df = pd.read_csv(csv_path)
    # Filter Member C models or use available models
    wanted = ["LogisticRegression", "RandomForest", "DecisionTree", "GaussianNB"]
    df_c = df[df.model.isin(wanted)].copy()

    if df_c.empty:
        raise RuntimeError("No Member C model rows found in experiments.csv.")

    df_c.to_csv(RES / "member_c_model_comparison.csv", index=False)

    # 1. Generate comparison plots
    plots = [
        ("cv_macro_f1_mean", "CV Candidate Macro F1 (5-fold Mean ± Std)", "cv_macro_f1.png"),
        ("student_test_top1_accuracy", "Student-test Observation Top-1 Accuracy", "student_test_top1_accuracy.png"),
        ("candidate_macro_f1", "Student-test Candidate Macro F1", "student_test_macro_f1.png"),
        ("candidate_log_loss", "Student-test Candidate Log Loss (lower is better)", "student_test_log_loss.png"),
        ("training_seconds", "Final Training Time (seconds)", "training_time.png"),
        ("inference_seconds_total", "Student-test Inference Time (seconds)", "inference_time.png"),
        ("route_continuity", "Predicted Route Directed Continuity", "route_continuity.png"),
    ]

    for col, title, filename in plots:
        if col not in df_c.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vals = df_c[col]
        bars = ax.bar(df_c.model, vals, color="#2b5c8f", alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        if col == "cv_macro_f1_mean" and "cv_macro_f1_std" in df_c.columns:
            ax.errorbar(
                range(len(df_c)),
                vals,
                yerr=df_c.cv_macro_f1_std,
                fmt="none",
                capsize=5,
                color="black",
                linewidth=1.2,
            )

        # Label values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(PLOTS / filename, dpi=170)
        plt.close(fig)

    # 2. Write report/member_c_results.md
    lines = [
        "# Member C — Model Experimentation and Evaluation Results",
        "",
        "## Experimental Setup",
        "The models were trained on the candidate features from the 80 training trips.",
        "Candidate rows were sampled by complete GPS-observation groups: at most 250 observations",
        "per training trip, retaining all candidate road arcs for each selected observation.",
        "The 10 student-testing trips were evaluated without observation subsampling.",
        "Five-fold GroupKFold cross-validation was used, grouping by trip ID to strictly guarantee zero cross-trip leakage.",
        "",
        "Candidate Macro F1 and Log Loss are candidate-row metrics. Observation Top-1 accuracy",
        "selects the candidate with the highest predicted positive probability for each GPS fix;",
        "a GPS fix whose reference arc is absent from its candidate set counts as incorrect.",
        "Route continuity is the fraction of consecutive predicted directed edges where the destination node of edge t equals the source node of edge t+1.",
        "",
        "## 3.1 Consolidated Comparison Table",
        "| Model | Family | CV Macro F1 | Student Top-1 | Student Candidate Macro F1 | Student Obs Macro F1 | Log Loss | Train s | Infer s | Continuity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in df_c.iterrows():
        cv_str = f"{r.cv_macro_f1_mean:.4f} ± {r.get('cv_macro_f1_std', 0.0):.4f}"
        lines.append(
            f"| {r.model} | {r.algorithm_family} | {cv_str} | {r.student_test_top1_accuracy:.4f} | "
            f"{r.candidate_macro_f1:.4f} | {r.get('student_test_observation_macro_f1', 0.0):.4f} | "
            f"{r.candidate_log_loss:.4f} | {r.training_seconds:.3f} | {r.inference_seconds_total:.3f} | {r.route_continuity:.4f} |"
        )

    lines += [
        "",
        "## 3.2 Individual Model Analysis",
        "",
        "### Logistic Regression",
        "- **Family**: Linear probabilistic classifier.",
        "- **Suitability & Strengths**: Fast, interpretable linear baseline. Median imputation and StandardScaler applied within cross-validation pipeline to prevent leakage.",
        "- **Limitations**: Linear decision boundary cannot capture non-linear geometric and topological interactions.",
        "",
        "### Random Forest",
        "- **Family**: Tree ensemble (bagging).",
        "- **Suitability & Strengths**: Captures complex non-linear interactions across spatial distances, bearings and node degrees. Robust against overfitting via bootstrap aggregation.",
        "- **Limitations**: Higher computational inference cost per candidate compared to simple linear models.",
        "",
        "### Decision Tree",
        "- **Family**: Tree-based classifier.",
        "- **Suitability & Strengths**: Directly models hierarchical splitting rules on continuous features; scale-invariant.",
        "- **Limitations**: Prone to variance and sensitive to small variations in training sample coordinates.",
        "",
        "## 3.3 Visualisations",
        "The generated plots are committed to `results/comparison_plots/`:",
        "- `cv_macro_f1.png`: 5-fold cross-validation candidate macro F1.",
        "- `student_test_top1_accuracy.png`: Observation-level Top-1 accuracy.",
        "- `student_test_macro_f1.png`: Candidate-level macro F1 on student-test.",
        "- `student_test_log_loss.png`: Candidate-level log loss.",
        "- `training_time.png` & `inference_time.png`: Wall-clock execution benchmarks.",
        "- `route_continuity.png`: Proportion of connected consecutive predicted edges.",
        "",
        "Confusion matrices are saved under `results/confusion_matrices/`.",
    ]

    (REP / "member_c_results.md").write_text("\n".join(lines), encoding="utf-8")

    # Update report/sections/3_1_model_comparison.md
    sec_3_1 = [
        "## 3.1 Model Comparison Table",
        "",
        "| Model | Family | Key Config | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student Macro F1 | Log Loss | Inference Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_c.iterrows():
        cv_str = f"{r.cv_macro_f1_mean:.4f} ± {r.get('cv_macro_f1_std', 0.0):.4f}"
        sec_3_1.append(
            f"| {r.model} | {r.algorithm_family} | 5-fold CV, group by trip | {cv_str} | {r.student_test_top1_accuracy:.4f} | "
            f"{r.route_continuity:.4f} | {r.candidate_macro_f1:.4f} | {r.candidate_log_loss:.4f} | {r.inference_seconds_total:.3f} |"
        )
    (REP_SECTIONS / "3_1_model_comparison.md").write_text("\n".join(sec_3_1) + "\n", encoding="utf-8")

    # Update report/sections/3_3_visualisations.md
    sec_3_3 = [
        "## 3.3 Comparison Visualisations",
        "",
        "The comparison figures show model metrics evaluated on the 5-fold GroupKFold cross-validation and the student-test partition:",
        "",
        "- **CV Macro F1**: `results/comparison_plots/cv_macro_f1.png`",
        "- **Top-1 Accuracy**: `results/comparison_plots/student_test_top1_accuracy.png`",
        "- **Candidate Macro F1**: `results/comparison_plots/student_test_macro_f1.png`",
        "- **Log Loss**: `results/comparison_plots/student_test_log_loss.png`",
        "- **Execution Benchmarks**: `results/comparison_plots/training_time.png`, `results/comparison_plots/inference_time.png`",
        "- **Route Continuity**: `results/comparison_plots/route_continuity.png`",
        "",
        "Candidate confusion matrices for each model are stored under `results/confusion_matrices/`.",
    ]
    (REP_SECTIONS / "3_3_visualisations.md").write_text("\n".join(sec_3_3) + "\n", encoding="utf-8")

    print("Wrote results/member_c_model_comparison.csv, plots, and report files successfully.")


if __name__ == "__main__":
    main()
