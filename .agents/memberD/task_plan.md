# Member D: Task Plan and Implementation Checklist

**Owner:** Member D  
**Models:** LightGBM (boosting), MLP (neural network)  
**Phase:** Bayesian hyperparameter optimization on winning model

## Phase 1: Model Wrappers (Priority: HIGH)

### 1.1 LightGBM Wrapper
- [ ] Implement `models/lgbm/model.py` with:
  - [ ] `fit()` with early stopping on validation fold
  - [ ] `predict_proba()` for probability predictions
  - [ ] `save()` and `load()` for model persistence
  - [ ] Group-aware settings (sample_weight by trip)
  - [ ] Training and inference timing
- [ ] Configure `configs/models/lgbm.yaml` with initial hyperparameters
- [ ] Test on sample of 5 trips from features v0

### 1.2 MLP Wrapper
- [ ] Implement `models/mlp/model.py` with:
  - [ ] `fit()` with epoch-wise training
  - [ ] `predict_proba()` for probability predictions
  - [ ] `save()` and `load()` for model persistence
  - [ ] **Save training/validation loss history per epoch** (CSV in results/)
  - [ ] StandardScaler integration (fit on training only)
  - [ ] Training and inference timing
- [ ] Configure `configs/models/mlp.yaml` with initial hyperparameters
- [ ] Test on sample of 5 trips from features v0
- [ ] Generate loss curve plots (ONLY model with epoch histories)

## Phase 2: Model Training and Comparison (Priority: HIGH)

### 2.1 Training Pipeline Integration
- [ ] Update `pipelines/04_train_compare.py` to include LightGBM and MLP
- [ ] Ensure both models use shared evaluation harness (Member C's code)
- [ ] Run 5-fold GroupKFold CV on 80 training trips
- [ ] Evaluate on 10 student-test trips
- [ ] Log runs to `experiments/runs/` with proper JSON schema

### 2.2 Calibration Checks
- [ ] Check calibration for both models (reliability plot, log loss)
- [ ] If poorly calibrated, fit isotonic/sigmoid calibration
- [ ] Keep calibration only if log loss improves in CV
- [ ] Record calibration results in run JSON

### 2.3 Model Comparison
- [ ] Compare LightGBM and MLP with other 3 models (jointly with Member C)
- [ ] Record CV macro F1, stability across folds, log loss, Top-1 accuracy
- [ ] Participate in Gate G2: winner selection by agreed metrics

## Phase 3: Bayesian Optimization (Priority: MEDIUM - after winner selected)

### 3.1 Search Space Definition
- [ ] Define justified search space in `configs/optuna.yaml`
- [ ] Document reasoning for each hyperparameter range
- [ ] Ensure search space is model-agnostic or adaptable to winner

### 3.2 Optuna Implementation
- [ ] Implement `hpo/optuna_search.py`:
  - [ ] TPE sampler
  - [ ] GroupKFold on 80 training trips (5 folds)
  - [ ] Objective: CV macro F1
  - [ ] Trial logging to `hpo/studies/` and `results/optuna_trials.csv`
- [ ] Implement `pipelines/05_tune_optuna.py`:
  - [ ] Run 30-50 trials
  - [ ] Save study database to `hpo/studies/study.db`
  - [ ] Save trial log to `hpo/studies/trials.csv`

### 3.3 Best Model Refit and Evaluation
- [ ] Refit winner with best parameters on all 80 training trips
- [ ] Evaluate once on student-test trips
- [ ] Compare tuned vs untuned:
  - [ ] CV macro F1
  - [ ] Student-test Top-1 accuracy
  - [ ] Student-test macro F1
  - [ ] Log loss
  - [ ] Inference time
- [ ] Report computational cost of search

## Phase 4: Artifacts Export (Priority: MEDIUM)

### 4.1 Export Pipeline
- [ ] Implement `pipelines/07_export_artifacts.py`:
  - [ ] Final tuned model
  - [ ] Fitted preprocessing state (scaler, etc.)
  - [ ] HMM config
  - [ ] Features version
  - [ ] Git commit hash
- [ ] Write `artifacts/README.md` with artifact descriptions

## Phase 5: Documentation (Priority: MEDIUM)

### 5.1 Feature Engineering Documentation
- [ ] Create `docs/feature_engineering.md`:
  - [ ] Feature categories and descriptions
  - [ ] Feature selection techniques
  - [ ] Preprocessing pipeline
  - [ ] How features are used in training

### 5.2 Report Sections
- [ ] Draft `report/sections/3_2_lightgbm.md`:
  - [ ] Why LightGBM suits PathSnap
  - [ ] Hyperparameters and initial values
  - [ ] Student-test performance
  - [ ] Overfitting/underfitting evidence
- [ ] Draft `report/sections/3_2_mlp.md`:
  - [ ] Why MLP suits PathSnap
  - [ ] Hyperparameters and initial values
  - [ ] Training/validation loss curves
  - [ ] Student-test performance
- [ ] Draft `report/sections/4_bayesian_optimisation.md`:
  - [ ] Search space justification
  - [ ] Objective function
  - [ ] Trial results
  - [ ] Tuned vs untuned comparison
  - [ ] Computational cost

## Dependencies and Coordination

### Blocked By (Critical Path)
- Member B: Features v0 handoff (`data/features/features.parquet`)
- Member C: Shared evaluation harness (`evaluation/metrics.py`, `evaluation/cv.py`)
- Member C: Run logger schema (`experiments/README.md`)

### Must Coordinate With
- Member C: Model comparison and winner selection (Gate G2)
- Member B: Viterbi decoder integration (after winner selected)

## Technical Decisions

### LightGBM Settings
- Early stopping: Use CV fold validation
- Group-aware: Not directly supported, but stratify by trip distribution
- Initial hyperparameters:
  - `num_leaves`: 31
  - `max_depth`: -1 (no limit)
  - `learning_rate`: 0.1
  - `n_estimators`: 100
  - `objective`: binary
  - `metric`: binary_logloss

### MLP Settings
- Architecture: 2 hidden layers (64, 32 neurons)
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.001
- Max epochs: 100
- Early stopping: patience=10
- Batch size: 256
- StandardScaler: fit on training data only

### Optuna Settings
- Sampler: TPE (Tree-structured Parzen Estimator)
- N trials: 40
- Objective: CV macro F1 (maximize)
- Pruning: MedianPruner
- Seed: from `configs/partitions.yaml`

## File Structure

```
models/
├── lgbm/
│   ├── __init__.py
│   └── model.py
├── mlp/
│   ├── __init__.py
│   └── model.py
└── factory.py (Member C)

hpo/
├── __init__.py
├── optuna_search.py
└── studies/
    ├── study.db
    └── trials.csv

configs/
├── models/
│   ├── lgbm.yaml
│   └── mlp.yaml
└── optuna.yaml

pipelines/
├── 04_train_compare.py (update)
├── 05_tune_optuna.py
└── 07_export_artifacts.py

results/
└── mlp_loss_history.csv

docs/
└── feature_engineering.md
```

## Metrics and Success Criteria

### Primary Metrics (for winner selection)
- CV macro F1 (5-fold, GroupKFold)
- Log loss (for Viterbi compatibility)
- Student-test Top-1 accuracy

### Secondary Metrics
- Training time
- Inference time
- Model complexity
- Calibration quality

## Timeline Estimate

- **Phase 1**: 3-4 hours (model wrappers)
- **Phase 2**: 2-3 hours (training and comparison)
- **Phase 3**: 2-3 hours (Bayesian optimization)
- **Phase 4**: 1 hour (artifact export)
- **Phase 5**: 2-3 hours (documentation)

**Total**: ~10-14 hours

## Notes

- All training must use only 80 training trips
- Student-test used only for development comparisons, not hyperparameter tuning
- No evaluator data in any pipeline
- Every run must produce a JSON in `experiments/runs/`
- MLP is the only model with loss curves (epoch-wise histories)
