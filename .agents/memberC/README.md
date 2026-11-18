# Member C: Implementation Status

**Owner:** Member C  
**Area:** `src/pathsnap/evaluation/`, `src/pathsnap/utils/`, `models/logreg/`, `models/dtree/`, `models/rf/`, `models/factory.py`, `report/sections/3_1_model_comparison.md`, `report/sections/3_3_visualisations.md`, `tests/test_metrics.py`  
**Last Updated:** November 2026

---

## ✅ Completed Deliverables

### Phase 1: Shared Evaluation Harness & Utilities
- [x] **Evaluation Metrics (`src/pathsnap/evaluation/metrics.py`)**:
  - [x] Candidate Macro F1, Precision, Recall, Accuracy
  - [x] Binary Cross-Entropy Log Loss
  - [x] Observation-level Top-1 Accuracy (strictly penalizes missing reference candidates)
  - [x] Candidate Miss Rate & Candidate Coverage
  - [x] Observation-level Macro F1
  - [x] Directed Route Continuity
  - [x] Sequence exact match evaluation
- [x] **Group-Aware Cross-Validation (`src/pathsnap/evaluation/cv.py`)**:
  - [x] 5-fold `GroupKFold` split strictly by `trip_id`
  - [x] Zero-leakage assertion validating disjoint trip partitions between train and validation folds
  - [x] Per-fold train and validation metric tracking
- [x] **Experiment Run Logger (`src/pathsnap/evaluation/run_logger.py`)**:
  - [x] Standardized run logging conforming to schema in `experiments/README.md`
  - [x] Automatic git commit recording and artifact serialization
  - [x] Run loader for consolidating report tables from `experiments/runs/*.json`
- [x] **Shared Utilities (`src/pathsnap/utils/`)**:
  - [x] Random seed management across Python, NumPy, and PyTorch (`seeds.py`)
  - [x] Wall-clock timer context manager for training and inference benchmarking (`timing.py`)

### Phase 2: Model Wrappers & Model Factory
- [x] **Logistic Regression Wrapper (`models/logreg/model.py`)**:
  - [x] Standardized interface: `fit`, `predict_proba`, `predict`, `save`, `load`
  - [x] Internal median imputation and StandardScaler fit strictly on training folds
  - [x] Wall-clock timing tracked for training and inference
- [x] **Decision Tree Wrapper (`models/dtree/model.py`)**:
  - [x] Standardized interface with median imputation
  - [x] Gini impurity minimization and leaf constraint regularization
- [x] **Random Forest Wrapper (`models/rf/model.py`)**:
  - [x] Standardized ensemble interface with bagging and feature subsampling
- [x] **Model Factory (`models/factory.py`)**:
  - [x] Unified model loader mapping YAML configs / model keys to wrapper classes (`logreg`, `dtree`, `rf`, `lgbm`, `mlp`)

### Phase 3: Configurations & Partitioning
- [x] `configs/partitions.yaml`: Defined 80 training trip IDs, 10 student-test trip IDs, and evaluator partition placeholder
- [x] `configs/models/logreg.yaml`: Hyperparameters, preprocessing, and objective
- [x] `configs/models/dtree.yaml`: Hyperparameters, preprocessing, and objective
- [x] `configs/models/rf.yaml`: Hyperparameters, preprocessing, and objective
- [x] `requirements.txt`: Specified core library versions

### Phase 4: Model Training, Evaluation & Benchmarking
- [x] Runner script (`models/run_member_c.py`) and training scripts:
  - `models/train_logistic.py`
  - `models/train_random_forest.py`
  - `models/train_decision_tree.py`
  - `models/train_gaussian_nb.py`
- [x] Pipeline Stage 4 integration (`pipelines/04_train_compare.py`)
- [x] Artifacts saved in `artifacts/` (`.joblib` and `_config.json`)
- [x] Run records committed to `experiments/runs/` (`.json`)
- [x] Results CSVs generated in `results/` (`experiments.csv`, `member_c_model_comparison.csv`, fold CSVs)

### Phase 5: Technical Report & Visualisations
- [x] Comparison plots in `results/comparison_plots/`:
  - `cv_macro_f1.png`
  - `student_test_top1_accuracy.png`
  - `student_test_macro_f1.png`
  - `student_test_log_loss.png`
  - `training_time.png` & `inference_time.png`
  - `route_continuity.png`
- [x] Confusion matrix plots in `results/confusion_matrices/`
- [x] Report Section 3.1: Consolidated comparison table (`report/sections/3_1_model_comparison.md`)
- [x] Report Section 3.2: Individual model analysis for Models 1–3 (`report/sections/3_2_per_model.md`)
- [x] Report Section 3.3: Comparison visualisations summary (`report/sections/3_3_visualisations.md`)
- [x] Detailed Member C summary report (`report/member_c_results.md`)

### Phase 6: Unit Testing Suite
- [x] `tests/test_metrics.py`: 8 hand-computed verification test cases
- [x] `tests/test_cv.py`: 4 tests for GroupKFold trip-level splitting and leakage detection
- [x] `tests/test_models.py`: 4 tests covering wrappers, serialization, and factory mapping
- [x] `tests/test_run_logger.py`: Schema validation against `experiments/README.md`
- [x] `tests/test_utils.py`: Seed reproducibility and timer benchmarks
- [x] `models/test_eval_harness.py`: End-to-end smoke test on raw trip candidates

---

## 📊 Summary of Model Results (Student-Test)

| Model | Algorithm Family | 5-Fold CV Macro F1 | Student Top-1 | Student Candidate F1 | Log Loss | Train Time (s) | Infer Time (s) | Route Continuity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **LogisticRegression** | Linear | 0.7315 ± 0.0146 | 0.6717 | 0.7328 | 0.3477 | 0.390 | 0.041 | 0.2796 |
| **DecisionTree** | Tree-based | 0.7753 ± 0.0163 | 0.6764 | 0.7880 | 0.3973 | 0.315 | 0.030 | 0.2555 |
| **RandomForest** | Ensemble (Bagging) | 0.8069 ± 0.0144 | 0.6898 | 0.8155 | 0.2217 | 0.952 | 0.255 | 0.2725 |
| **GaussianNB** | Probabilistic | 0.7657 ± 0.0124 | 0.6173 | 0.7601 | 0.5214 | 0.210 | 0.051 | 0.2914 |
