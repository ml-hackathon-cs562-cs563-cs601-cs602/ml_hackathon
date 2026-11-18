## 3.1 Model Comparison Table

| Model | Family | Key Config | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student Macro F1 | Log Loss | Inference Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | linear probabilistic classifier | 5-fold CV, group by trip | 0.7315 ± 0.0146 | 0.6717 | 0.2796 | 0.7328 | 0.3477 | 0.041 |
| RandomForest | tree ensemble (bagging) | 5-fold CV, group by trip | 0.8069 ± 0.0144 | 0.6898 | 0.2725 | 0.8155 | 0.2217 | 0.255 |
| DecisionTree | tree-based | 5-fold CV, group by trip | 0.7753 ± 0.0163 | 0.6764 | 0.2555 | 0.7880 | 0.3973 | 0.030 |
| GaussianNB | probabilistic generative classifier | 5-fold CV, group by trip | 0.7657 ± 0.0124 | 0.6173 | 0.2914 | 0.7601 | 0.5214 | 0.051 |
