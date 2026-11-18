# Models 4 & 5: LightGBM and MLP

**Owner:** Member D  
**Last Updated:** November 2026  
**Version:** v0 (untuned baseline)

---

## Overview

This document describes the two models implemented by Member D for the PathSnap map-matching project:

1. **Model 4: LightGBM** - Gradient boosting with leaf-wise tree growth
2. **Model 5: MLP** - Multi-layer perceptron neural network with epoch-wise loss tracking

Both models are binary classifiers that predict the probability that a candidate arc is the correct match for a GPS observation point. These probabilities feed into the HMM/Viterbi decoder for sequence-level route reconstruction.

---

## Model 4: LightGBM

### 1. Why LightGBM?

**Algorithm Family:** Gradient Boosted Decision Trees (GBDT)

**Why I chose LightGBM for PathSnap:**

| Reason | Explanation |
|--------|-------------|
| **High accuracy on tabular data** | LightGBM consistently outperforms other methods on structured feature tables like our candidate table |
| **Fast training and inference** | Leaf-wise growth strategy and histogram-based algorithms make it significantly faster than XGBoost and Random Forest, critical for API latency requirements |
| **Handles imbalanced data well** | Our candidate table has ~5% positive rate (true arcs are rare among K candidates). LightGBM's `scale_pos_weight` and built-in class weighting handle this naturally |
| **No feature scaling required** | Tree-based models don't require StandardScaler, simplifying the pipeline and reducing preprocessing errors |
| **Feature importance built-in** | Gain and split importance help understand which geometric/contextual features matter most |
| **Robust to outliers** | GPS noise creates outliers in distance and heading features; trees handle these better than linear models |
| **Calibrated probabilities** | While not perfectly calibrated out of the box, LightGBM probabilities are smoother than Random Forest and respond well to Platt scaling or isotonic regression |

### 2. Advantages of LightGBM

| Advantage | Benefit for PathSnap |
|-----------|---------------------|
| **Speed** | Training on 80 trips (~2-4M candidate rows) completes in ~90 seconds; inference on 10 student trips (~150K rows) in ~1.1 seconds. Critical for real-time API |
| **Memory efficient** | Histogram-based algorithm uses less memory than pre-sorted split finding, allowing larger batch sizes |
| **Leaf-wise growth** | Grows trees by splitting the leaf with maximum delta loss (not level-wise). Achieves lower loss with fewer trees, improving both accuracy and speed |
| **Built-in categorical support** | Can handle categorical features natively (though our current features are all numeric) |
| **Early stopping** | Monitors validation loss and stops training when no improvement, preventing overfitting and reducing training time |
| **Parallel and GPU training** | Can leverage multiple cores and GPUs if available in the deployment environment |

### 3. Disadvantages and Mitigations

| Disadvantage | Mitigation |
|--------------|------------|
| **Prone to overfitting on small data** | Used early stopping with 30 rounds patience; limited `num_leaves` to 31; set `min_data_in_leaf=20` |
| **Many hyperparameters** | Started with conservative defaults from `configs/models/lgbm.yaml`; will tune only the winner model via Optuna |
| **Leaf-wise growth can overfit** | Limited tree depth (`max_depth=-1` lets leaves control complexity) and used bagging fraction |
| **Black box** | Compensated with feature importance analysis; will document gain-based importance in report |

### 4. Architecture and Hyperparameters

#### 4.1 Model Configuration

```yaml
# From configs/models/lgbm.yaml
objective: binary
metric: binary_logloss
boosting_type: gbdt

# Learning
learning_rate: 0.1
num_leaves: 31
max_depth: -1  # No explicit limit, controlled by num_leaves
min_data_in_leaf: 20

# Regularization
feature_fraction: 0.8    # Randomly select 80% of features per tree
bagging_fraction: 0.8    # Randomly select 80% of data per tree
bagging_freq: 1          # Enable bagging
lambda_l1: 0.0           # L1 regularization (disabled initially)
lambda_l2: 0.0           # L2 regularization (disabled initially)

# Reproducibility
seed: 42
bagging_seed: 42
feature_fraction_seed: 42
deterministic: true
num_threads: 0  # Use all available cores
verbosity: -1
```

#### 4.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_rounds` | 1000 | Upper limit; early stopping typically stops at ~60-80 rounds |
| `early_stopping_rounds` | 30 | Stop if no improvement in validation log loss for 30 rounds |
| `validation_split` | 15% of training-fold trips (inner_val_frac) | Held-out training trips for early stopping, NOT the CV validation fold |
| `n_folds` | 5 | GroupKFold by trip to prevent leakage |

#### 4.3 Why These Hyperparameters?

| Hyperparameter | Reasoning |
|----------------|-----------|
| `learning_rate=0.1` | Moderate learning rate balances training speed and convergence. Lower rates (0.01) would require more trees; higher rates (0.3) risk overshooting |
| `num_leaves=31` | Default value. $2^{\text{max_depth}}$ for depth-5 trees = 32 leaves, so this is equivalent to depth ~5. Prevents over-complex trees |
| `min_data_in_leaf=20` | Prevents splits that only affect a few samples, reducing overfitting to noise |
| `feature_fraction=0.8` | Random feature selection per tree adds randomness, reduces overfitting, and speeds up training |
| `bagging_fraction=0.8` | Random data sampling per iteration improves generalization |
| `early_stopping=30` | Patient enough to find optimum but not waste time on plateaued training |

### 5. Cross-Validation Results

**5-fold GroupKFold on 80 training trips:**

| Metric | Mean ± Std | Min → Max |
|--------|-----------|-----------|
| **Candidate Macro F1** | 0.9042 ± 0.0148 | (fold stability: 0.8894 → 0.9190) |
| **Candidate Precision** | 0.8502 ± 0.0357 | |
| **Candidate Recall** | 0.8141 ± 0.0252 | |
| **Candidate Accuracy** | 0.9592 ± 0.0067 | |
| **Candidate Log Loss** | 0.1040 ± 0.0149 | Critical for Viterbi decoder |
| **Top-1 Accuracy** | 0.8014 ± 0.0729 | Observation-level Top-1 |
| **Best Iteration** | 62.4 ± 9.4 rounds | Early stopping worked |
| **Train Macro F1** | 0.9213 ± 0.0079 | Slightly higher than CV (expected) |
| **Train Log Loss** | 0.0870 ± 0.0063 | Lower than CV (expected) |
| **Train Time** | 17.6 ± 1.9 seconds | Per fold |
| **Inference Time** | 1.15 ± 0.23 seconds | Per fold |

#### 5.1 Overfitting Analysis

| Metric | Train | CV (Validation) | Gap |
|--------|-------|-----------------|-----|
| Macro F1 | 0.9213 | 0.9042 | +0.0171 |
| Log Loss | 0.0870 | 0.1040 | +0.0170 |

**Interpretation:** 
- Small positive gap indicates mild overfitting (expected)
- Train-CV F1 gap of 1.7% is acceptable for a boosting model
- Log loss gap of 0.017 is within normal range
- Early stopping at ~62 rounds (well below 1000 limit) shows the model found a good optimum

#### 5.2 Fold Stability

- **Macro F1 range:** 0.8894 → 0.9190 (spread = 0.0296)
- **Standard deviation:** 0.0148 (1.6% of mean)
- **Interpretation:** Folds are reasonably stable; no catastrophic failures on any fold

### 6. Student-Test Evaluation

**Final model refit on all 80 training trips:**

| Metric | Value |
|--------|-------|
| **Candidate Macro F1** | TBD (from final run) |
| **Candidate Log Loss** | TBD |
| **Top-1 Accuracy** | TBD |
| **Train Time** | TBD |
| **Inference Time** | TBD |

*(Run the final evaluation cell in the notebook to populate these values)*

### 7. Feature Importance

*(To be generated after training)*

Top features by gain:
1. `dist_m` - Perpendicular distance (expected: strongest signal)
2. `distance_rank` - Relative position among candidates
3. `heading_diff` - Angular mismatch
4. `prev_dist_to_same_arc` - Temporal continuity
5. `next_dist_to_same_arc` - Temporal continuity

### 8. Calibration

*(To be populated from calibration check)*

| Method | Mean Log Loss | Kept? |
|--------|---------------|-------|
| Raw | TBD | TBD |
| Sigmoid (Platt) | TBD | TBD |
| Isotonic | TBD | TBD |

---

## Model 5: MLP (Multi-Layer Perceptron)

### 1. Why MLP?

**Algorithm Family:** Feedforward Neural Network

**Why I chose MLP for PathSnap:**

| Reason | Explanation |
|--------|-------------|
| **Learns non-linear relationships** | GPS-to-arc matching involves complex geometric relationships that linear models cannot capture. MLP's hidden layers can learn these interactions |
| **Different inductive bias from trees** | Tree models partition feature space by axis-aligned splits. Neural networks learn smooth, continuous decision boundaries. Ensembling diverse models often improves overall performance |
| **Feature interactions** | MLP can learn interactions between features automatically (e.g., distance AND heading together matter more than either alone) |
| **Epoch-wise training** | Unlike trees, MLP trains incrementally, allowing us to observe training and validation loss curves over epochs. This provides insight into convergence and overfitting |
| **Standard baseline** | MLP is a well-understood neural network baseline. If it performs well, it validates the feature engineering. If it fails, it suggests features need improvement |
| **Scalable** | Once trained, MLP inference is a series of matrix multiplications—very fast and parallelizable |

### 2. Advantages of MLP

| Advantage | Benefit for PathSnap |
|-----------|---------------------|
| **Smooth probability outputs** | Neural networks produce smooth, continuous probability surfaces, which are ideal for the HMM emission probabilities |
| **Fast inference** | After training, inference is just forward propagation: X → W₁ → ReLU → W₂ → ReLU → W₃ → Sigmoid. Very fast (~0.25s for 150K rows) |
| **Epoch-wise loss tracking** | Only MLP has training and validation loss curves. This helps diagnose overfitting, underfitting, and optimal stopping point |
| **Universal approximator** | Theoretically can approximate any continuous function given enough neurons and data |
| **Learns feature interactions** | Hidden layers can learn non-linear combinations of input features (e.g., "close distance AND matching heading") |
| **Consistent with deep learning trends** | If the project extends to larger datasets or more complex features (e.g., raw GPS sequences), MLP is a stepping stone to deeper architectures |

### 3. Disadvantages and Mitigations

| Disadvantage | Mitigation |
|--------------|------------|
| **Requires feature scaling** | StandardScaler fitted on training data only, applied to validation/test. Added to pipeline in `artifacts/scaler.joblib` |
| **Many hyperparameters** | Used conservative defaults from `configs/models/mlp.yaml`; limited to 2 hidden layers (64, 32) to prevent overfitting on moderate dataset size |
| **Can overfit easily** | Early stopping with patience=5 epochs; L2 regularization (alpha=0.0001); held-out validation trips for monitoring |
| **Black box** | Documented architecture and training curves; no interpretability tools used (future: SHAP, LIME) |
| **Sensitive to initialization** | Fixed random seed (42); warm_start=False for fresh initialization each fold |
| **Slower training than LightGBM** | Training took 134s vs LightGBM's 90s. Acceptable for offline training; inference is fast |

### 4. Architecture and Hyperparameters

#### 4.1 Network Architecture

```
Input Layer: 15 features (from Member B's pipeline)
    ↓
Hidden Layer 1: 64 neurons, ReLU activation
    ↓
Hidden Layer 2: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron, Sigmoid activation
```

**Architecture rationale:**

| Design Choice | Reasoning |
|---------------|-----------|
| **2 hidden layers** | Sufficient to learn non-linear relationships without being overly complex. Deeper networks would risk overfitting on ~2M training rows |
| **64, 32 neurons** | Gradual reduction (funnel architecture) forces the network to learn compact representations. 64 neurons can capture feature interactions; 32 refines the representation |
| **ReLU activation** | Standard choice for hidden layers. Avoids vanishing gradient problem of sigmoid/tanh. Computationally efficient |
| **Sigmoid output** | Binary classification requires probability in [0, 1]. Sigmoid is natural choice |
| **No dropout** | Early stopping provides regularization; dropout would add another hyperparameter to tune |

#### 4.2 Model Configuration

```yaml
# From configs/models/mlp.yaml
hidden_layer_sizes: [64, 32]
activation: relu
solver: adam

# Regularization
alpha: 0.0001  # L2 penalty

# Learning
learning_rate: constant
learning_rate_init: 0.001
batch_size: 256

# Training (manual epoch loop)
max_iter: 1  # We use partial_fit for epoch-wise tracking
warm_start: true
early_stopping: false  # We implement manual early stopping

# Reproducibility
random_state: 42
```

#### 4.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_epochs` | 100 | Upper limit; early stopping typically stops at ~10-15 epochs |
| `early_stop_patience` | 5 epochs | Stop if validation loss doesn't improve for 5 consecutive epochs |
| `early_stop_min_delta` | 1e-4 | Minimum improvement to count as progress |
| `batch_size` | 256 | Standard batch size. Larger batches converge faster but may generalize worse |
| `validation_split` | 15% of training-fold trips (inner_val_frac) | Held-out training trips for early stopping |
| `n_folds` | 5 | GroupKFold by trip |
| `optimizer` | Adam | Adaptive learning rate, good default for most problems |
| `learning_rate` | 0.001 (constant) | Default for Adam. Not tuned yet |

#### 4.4 Why These Hyperparameters?

| Hyperparameter | Reasoning |
|----------------|-----------|
| `hidden_layer_sizes=(64, 32)` | Moderate capacity for tabular data. Too many neurons → overfitting; too few → underfitting. Funnel shape encourages feature learning |
| `activation=relu` | Standard choice. Faster than tanh/sigmoid, avoids vanishing gradients |
| `solver=adam` | Adaptive learning rates for each parameter. Works well with default settings, less sensitive to learning rate than SGD |
| `alpha=0.0001` | Light L2 regularization. Prevents weights from growing too large, improves generalization |
| `batch_size=256` | Balance between speed (larger batches are faster) and generalization (smaller batches add noise) |
| `learning_rate_init=0.001` | Default for Adam. Will tune if this model is selected for optimization |
| `early_stop_patience=5` | Aggressive early stopping to prevent overfitting. Monitor validation loss on held-out training trips |

### 5. Cross-Validation Results

**5-fold GroupKFold on 80 training trips:**

| Metric | Mean ± Std |
|--------|-----------|
| **Candidate Macro F1** | TBD |
| **Candidate Log Loss** | TBD |
| **Top-1 Accuracy** | TBD |
| **Best Epoch** | TBD |
| **Train Time** | TBD |

*(Run CV to populate these values)*

### 6. Student-Test Evaluation

**Final model refit on all 80 training trips:**

| Metric | Value |
|--------|-------|
| **Candidate Macro F1** | 0.9107 |
| **Candidate Precision** | 0.8572 |
| **Candidate Recall** | 0.8303 |
| **Candidate Accuracy** | 0.9613 |
| **Candidate Log Loss** | 0.0975 |
| **Student-Test Top-1 Accuracy** | 0.8483 |
| **Student-Test Observation Macro F1** | 0.7377 |
| **Route Continuity** | 0.2222 |
| **Candidate Coverage** | 0.9972 |
| **Candidate Miss Rate** | 0.0028 |
| **Final Epochs** | 11 |
| **Train Time** | 134.1 seconds |
| **Inference Time** | 0.247 seconds |

#### 6.1 Key Observations

1. **Early stopping worked:** Training stopped at 11 epochs (well below 100 limit), indicating the model converged quickly
2. **Fast inference:** 0.247s for 152,072 rows = ~616K predictions/second. Excellent for API
3. **Top-1 verification:** C's function and fast version matched (0.848278), confirming implementation correctness
4. **Log loss:** 0.0975 is excellent (lower is better). Critical for well-calibrated Viterbi emissions
5. **Route continuity:** 0.2222 indicates ~22% of consecutive predictions are on connected arcs. Decoder tuning should improve this

### 7. Loss Curves

*(To be generated from training)*

The MLP saves epoch-wise training and validation loss to:
- `results/mlp_cv_loss_histories.csv` - Per-fold loss curves
- `results/mlp_final_loss_history.csv` - Final model loss curve
- `results/mlp_loss_curves.png` - Visualization

**Expected pattern:**
- Training loss decreases monotonically
- Validation loss decreases, then plateaus or increases slightly (early stopping point)
- Gap between train and validation loss indicates overfitting level

### 8. Feature Scaling

MLP requires scaled features. The pipeline:

```python
# Fitted on training-fold rows only
scaler = StandardScaler().fit(X_train[fit_idx])

# Applied to validation and test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Why StandardScaler?**
- Centers features at mean=0, scales to std=1
- Prevents features with large ranges (e.g., distance in meters) from dominating features with small ranges (e.g., rank)
- Required for neural networks using gradient descent

**Saved artifacts:**
- `artifacts/scaler.joblib` - Fitted StandardScaler
- `artifacts/mlp_feature_names.json` - Feature order (must match at inference)

### 9. Calibration

*(To be populated from calibration check)*

| Method | Mean Log Loss | Kept? |
|--------|---------------|-------|
| Raw | 0.0975 | TBD |
| Sigmoid (Platt) | TBD | TBD |
| Isotonic | TBD | TBD |

---

## Model Comparison: LightGBM vs MLP

### Performance Summary

| Metric | LightGBM (CV) | MLP (Student-Test) | Notes |
|--------|---------------|-------------------|-------|
| **Candidate Macro F1** | 0.9042 ± 0.0148 | 0.9107 | MLP slightly higher (different evaluation sets) |
| **Candidate Log Loss** | 0.1040 ± 0.0149 | 0.0975 | MLP lower (better for Viterbi) |
| **Top-1 Accuracy** | 0.8014 ± 0.0729 (CV) | 0.8483 (student-test) | Higher on student-test (expected) |
| **Train Time** | 17.6s per fold | 134.1s total | LightGBM faster |
| **Inference Time** | 1.15s per fold | 0.247s total | MLP faster per prediction |

**Important:** Direct comparison requires evaluating both models on the same test set (student-test). The CV metrics for LightGBM are on held-out training folds, while MLP metrics are on student-test trips.

### Training Characteristics

| Aspect | LightGBM | MLP |
|--------|----------|-----|
| **Training speed** | Fast (90s total) | Slower (134s total) |
| **Inference speed** | Fast (1.15s per fold) | Very fast (0.247s for 150K rows) |
| **Feature scaling** | Not required | Required (StandardScaler) |
| **Hyperparameters** | Many (trees, leaves, learning rate, etc.) | Many (layers, neurons, learning rate, batch size, etc.) |
| **Interpretability** | Feature importance (gain/split) | Black box (no built-in interpretability) |
| **Loss curves** | Per-iteration (boosting rounds) | Per-epoch (neural network training) |
| **Probability calibration** | Good (but benefits from Platt/isotonic) | Good (sigmoid output naturally calibrated) |

### Which Model to Choose?

**Winner selection criteria:**

1. **CV macro F1 and stability** - LightGBM: 0.9042 ± 0.0148; MLP: TBD
2. **Log loss** - MLP: 0.0975 (better); LightGBM: 0.1040
3. **Student-test Top-1** - MLP: 0.8483; LightGBM: TBD
4. **Inference speed** - MLP: 0.247s for 150K rows; LightGBM: ~1.15s per fold
5. **Deployment complexity** - LightGBM simpler (no scaler); MLP requires scaler

**Recommendation:**
- Run both models on student-test trips for fair comparison
- Check calibration (log loss matters for Viterbi)
- Consider inference speed for API requirements
- Winner will be tuned via Bayesian optimization

---

## Next Steps

### For Both Models

1. **Complete student-test evaluation** - Run final refit and evaluation cells
2. **Calibration check** - Fit Platt/isotonic, compare log loss
3. **Feature importance analysis** - For LightGBM, document top features
4. **Loss curve analysis** - For MLP, save and analyze training curves

### If LightGBM Wins

1. **Bayesian optimization** via Optuna:
   - Search space: `learning_rate`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`
   - Objective: CV macro F1 or log loss
   - Trials: 30-50
2. **Export artifacts:** Model, feature names, config

### If MLP Wins

1. **Bayesian optimization** via Optuna:
   - Search space: `hidden_layer_sizes`, `alpha`, `learning_rate_init`, `batch_size`
   - Objective: CV macro F1 or log loss
   - Trials: 30-50
2. **Export artifacts:** Model, scaler, feature names, config

### Integration with HMM

Both models output probabilities P(candidate is correct | features), which feed into:

1. **Emission probabilities:** log(P) + constant
2. **Viterbi decoder:** Finds most likely sequence of arcs
3. **Route output:** One arc per relevant GPS observation

---

## Artifacts

### LightGBM

| Artifact | Path | Description |
|----------|------|-------------|
| Model | `artifacts/lgbm_untuned.txt` | LightGBM model file |
| Feature names | `artifacts/lgbm_feature_names.json` | Feature order |
| CV results | `results/lgbm_cv_folds.csv` | Per-fold metrics |
| Feature importance | `results/lgbm_feature_importance.csv` | Gain-based importance |
| Calibration | `results/lgbm_calibration.csv` | Calibration comparison |
| Student-test results | `results/lgbm_student_test.csv` | Final evaluation |
| Run JSON | `experiments/runs/run_lightgbm_v0.json` | Complete run log |

### MLP

| Artifact | Path | Description |
|----------|------|-------------|
| Model | `artifacts/mlp_untuned.joblib` | Trained MLP classifier |
| Scaler | `artifacts/scaler.joblib` | Fitted StandardScaler |
| Feature names | `artifacts/mlp_feature_names.json` | Feature order |
| CV results | `results/mlp_cv_folds.csv` | Per-fold metrics |
| Loss histories | `results/mlp_cv_loss_histories.csv` | Per-fold loss curves |
| Final loss history | `results/mlp_final_loss_history.csv` | Final model loss curve |
| Loss curves plot | `results/mlp_loss_curves.png` | Visualization |
| Calibration | `results/mlp_calibration.csv` | Calibration comparison |
| Student-test results | `results/mlp_student_test.csv` | Final evaluation |
| Run JSON | `experiments/runs/run_mlp_v0.json` | Complete run log |

---

## Code Locations

| Component | Path |
|-----------|------|
| LightGBM config | `configs/models/lgbm.yaml` |
| MLP config | `configs/models/mlp.yaml` |
| LightGBM wrapper | `src/pathsnap/models/lgbm/model.py` |
| MLP wrapper | `src/pathsnap/models/mlp/model.py` |
| LightGBM notebook | `notebooks/model4_lightgbm.ipynb` |
| MLP notebook | `notebooks/model5_mlp.ipynb` |
| Optuna config | `configs/optuna.yaml` |

---

## References

1. **LightGBM Paper:** Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
2. **LightGBM Documentation:** https://lightgbm.readthedocs.io/
3. **MLP (sklearn):** https://scikit-learn.org/stable/modules/neural_networks_supervised.html
4. **Adam Optimizer:** Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"
5. **Early Stopping:** Prechelt, L. (1998). "Early Stopping - But When?"

---

**End of Document**
