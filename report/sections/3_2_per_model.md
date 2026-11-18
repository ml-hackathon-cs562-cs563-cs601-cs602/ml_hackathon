## 3.2 Individual Model Analysis

### Model 1: Logistic Regression
- **Algorithm family**: Linear probabilistic classifier.
- **Suitability**: Candidate scoring is a binary classification task on continuous distance, bearing, speed and topology features. Logistic regression provides a fast, interpretable linear baseline and calibrated posterior probabilities.
- **Strengths and limitations**: Extremely fast to fit and infer. However, linear decision boundaries cannot capture non-linear interactions (e.g. geometric distance vs bearing angle).
- **Hyperparameters & Preprocessing**: `C=1.0`, `solver=lbfgs`, `max_iter=500`, `class_weight=balanced`. Pipeline incorporates `SimpleImputer(strategy='median')` and `StandardScaler` fitted strictly on training folds.
- **Objective / Loss**: Binary logistic cross-entropy loss with L2 regularization.
- **Performance**: 5-fold CV candidate Macro F1: `0.7315 ± 0.0146`. Student-test Top-1 accuracy: `0.6717`, student-test candidate Macro F1: `0.7328`, log loss: `0.3477`.
- **Fit diagnostics**: Low overfitting gap (train-minus-CV gap = `0.0013`). Error is dominated by structural bias rather than variance.

---

### Model 2: Random Forest
- **Algorithm family**: Tree ensemble (bagging).
- **Suitability**: Effectively captures non-linear interactions between GPS dynamics and road network geometry without requiring feature normalization.
- **Strengths and limitations**: Strong generalization and robust against outlier candidate features. Higher inference overhead and memory usage due to ensemble size.
- **Hyperparameters & Preprocessing**: `n_estimators=20`, `max_depth=10`, `min_samples_split=4`, `min_samples_leaf=5`, `max_features='sqrt'`, `max_samples=0.5`, `class_weight='balanced_subsample'`, `bootstrap=True`.
- **Objective / Loss**: Gini impurity minimization at individual node splits.
- **Performance**: 5-fold CV candidate Macro F1: `0.8069 ± 0.0144`. Student-test Top-1 accuracy: `0.6898`, student-test candidate Macro F1: `0.8155`, log loss: `0.2217`.
- **Fit diagnostics**: Controlled overfitting via bagging and tree depth constraints (train-minus-CV gap: `0.0209`).

---

### Model 3: Decision Tree
- **Algorithm family**: Tree-based classifier.
- **Suitability**: Provides explicit hierarchical decision rules for road edge candidate filtering and candidate rank splitting.
- **Strengths and limitations**: Highly interpretable and scale-invariant; prone to high variance on noisy GPS fixes if unconstrained.
- **Hyperparameters & Preprocessing**: `max_depth=10`, `min_samples_split=4`, `min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42`.
- **Objective / Loss**: Gini impurity minimization.
- **Performance**: 5-fold CV candidate Macro F1: `0.7753 ± 0.0163`. Student-test Top-1 accuracy: `0.6764`, student-test candidate Macro F1: `0.7880`, log loss: `0.3973`.
- **Fit diagnostics**: Moderate train-minus-CV gap, bounded by minimum leaf size and depth limits.

---

### Model 4: LightGBM (Member D)
*Documented by Member D.*

---

### Model 5: MLP (Member D)
*Documented by Member D.*
