# Member C — Model Experimentation and Evaluation Results

## Experimental Setup
The models were trained on the candidate features from the 80 training trips.
Candidate rows were sampled by complete GPS-observation groups: at most 250 observations
per training trip, retaining all candidate road arcs for each selected observation.
The 10 student-testing trips were evaluated without observation subsampling.
Five-fold GroupKFold cross-validation was used, grouping by trip ID to strictly guarantee zero cross-trip leakage.

Candidate Macro F1 and Log Loss are candidate-row metrics. Observation Top-1 accuracy
selects the candidate with the highest predicted positive probability for each GPS fix;
a GPS fix whose reference arc is absent from its candidate set counts as incorrect.
Route continuity is the fraction of consecutive predicted directed edges where the destination node of edge t equals the source node of edge t+1.

## 3.1 Consolidated Comparison Table
| Model | Family | CV Macro F1 | Student Top-1 | Student Candidate Macro F1 | Student Obs Macro F1 | Log Loss | Train s | Infer s | Continuity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | linear probabilistic classifier | 0.7315 ± 0.0146 | 0.6717 | 0.7328 | 0.5051 | 0.3477 | 0.390 | 0.041 | 0.2796 |
| RandomForest | tree ensemble (bagging) | 0.8069 ± 0.0144 | 0.6898 | 0.8155 | 0.5237 | 0.2217 | 1.144 | 0.255 | 0.2725 |
| DecisionTree | tree-based | 0.7753 ± 0.0163 | 0.6764 | 0.7880 | 0.4851 | 0.3973 | 2.541 | 0.030 | 0.2555 |
| GaussianNB | probabilistic generative classifier | 0.7657 ± 0.0124 | 0.6173 | 0.7601 | 0.4711 | 0.5214 | 0.222 | 0.051 | 0.2914 |

## 3.2 Individual Model Analysis

### Logistic Regression
- **Family**: Linear probabilistic classifier.
- **Suitability & Strengths**: Fast, interpretable linear baseline. Median imputation and StandardScaler applied within cross-validation pipeline to prevent leakage.
- **Limitations**: Linear decision boundary cannot capture non-linear geometric and topological interactions.

### Random Forest
- **Family**: Tree ensemble (bagging).
- **Suitability & Strengths**: Captures complex non-linear interactions across spatial distances, bearings and node degrees. Robust against overfitting via bootstrap aggregation.
- **Limitations**: Higher computational inference cost per candidate compared to simple linear models.

### Decision Tree
- **Family**: Tree-based classifier.
- **Suitability & Strengths**: Directly models hierarchical splitting rules on continuous features; scale-invariant.
- **Limitations**: Prone to variance and sensitive to small variations in training sample coordinates.

## 3.3 Visualisations
The generated plots are committed to `results/comparison_plots/`:
- `cv_macro_f1.png`: 5-fold cross-validation candidate macro F1.
- `student_test_top1_accuracy.png`: Observation-level Top-1 accuracy.
- `student_test_macro_f1.png`: Candidate-level macro F1 on student-test.
- `student_test_log_loss.png`: Candidate-level log loss.
- `training_time.png` & `inference_time.png`: Wall-clock execution benchmarks.
- `route_continuity.png`: Proportion of connected consecutive predicted edges.

Confusion matrices are saved under `results/confusion_matrices/`.