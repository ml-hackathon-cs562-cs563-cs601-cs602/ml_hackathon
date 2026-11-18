# 4. Bayesian Optimisation

## 4.1 Search Space, Ranges, and Objective
Hyperparameter optimisation for the candidate edge classifier was performed using Optuna's Tree-structured Parzen Estimator (TPE). To respect the evaluation constraints, optimisation was conducted strictly on the 80 training trips using 5-fold GroupKFold cross-validation grouped by `trip_id` (preventing any trip data leakage between folds).

The objective function maximised the out-of-fold validation Macro F1 score across candidate edge classifications.

| Hyperparameter | Distribution / Scale | Range | Default / Untuned | Best Tuned Value |
| :--- | :--- | :--- | :--- | :--- |
| `n_estimators` | Integer uniform | [100, 500] | 200 | 400 |
| `learning_rate` | Log-uniform float | [0.01, 0.20] | 0.10 | 0.03364 |
| `max_depth` | Integer uniform | [4, 12] | 6 | 10 |
| `min_child_weight` | Log-uniform float | [1.0, 50.0] | 1.0 | 46.735 |
| `subsample` | Float uniform | [0.50, 1.00] | 1.0 | 0.8847 |
| `colsample_bytree` | Float uniform | [0.50, 1.00] | 1.0 | 0.6420 |
| `reg_lambda` | Log-uniform float | [1e-3, 10.0] | 1.0 | 0.00156 |
| `gamma` | Float uniform | [0.0, 5.0] | 0.0 | 1.8453 |

A total of 20 trials were evaluated across 1,033 seconds (~17.2 minutes).

## 4.2 Tuned vs Untuned Model Comparison
All evaluations were computed using the strict monorepo partition scheme (`configs/partitions.yaml`), evaluating on the 10 student-test trips.

| Metric | Untuned XGBoost | Tuned XGBoost | Delta |
| :--- | :--- | :--- | :--- |
| **CV Macro F1** | 0.9126 | 0.9132 | +0.0006 |
| **CV Top-1 Accuracy** | 0.8486 | 0.8483 | -0.0003 |
| **Test Top-1 Accuracy** | 0.8557 | 0.8540 | -0.0017 |
| **Test Macro F1** | 0.9149 | 0.9144 | -0.0005 |
| **Test Log Loss** | 0.0908 | 0.0901 | -0.0007 (improved) |
| **Test Route Continuity** | 0.7539 | 0.7530 | -0.0009 |
| **Training Time (s)** | 10.34 s | 25.73 s | +15.39 s |
| **Inference Time (s)** | 0.160 s | 0.780 s | +0.620 s |

![Optuna Optimisation History](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/optuna_history.png)
*Figure 4.1: Optuna Bayesian optimisation history showing objective value progression across trials.*

![Overfitting Check](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/overfit_check.png)
*Figure 4.2: Train vs validation performance check across iterations.*

The tuning process calibrated class probabilities, reducing cross-entropy log loss from 0.0908 to 0.0901, while adding regularization (`gamma=1.845`, `min_child_weight=46.74`) to prevent over-specialization on dominant road topologies.
