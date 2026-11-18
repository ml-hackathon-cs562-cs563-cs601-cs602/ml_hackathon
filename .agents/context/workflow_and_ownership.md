# Workflow and ownership

## Owners
| Member | Area | Phase 1 | Phase 2 |
|---|---|---|---|
| A | api/, ui/, viz/, ingest/, report sections 2.2, 6, 7 | EDA, data loader, Folium maps | FastAPI, UI, README, deployment |
| B | candidates/, features/, labels/, geo/, hmm/, pipelines 02, 03, 06 | Candidates and features v0 then v1 (critical path) | Viterbi, decoder tuning, HMM comparison |
| C | evaluation/, utils/, models/logreg, dtree, rf, report 3.1 and 3.3 | Shared harness | Logistic Regression, Decision Tree, Random Forest |
| D | models/lgbm, models/mlp, hpo/, pipelines 05, 07, report section 4 | LightGBM, MLP | Optuna tuning on the winner, export |

Members must be able to explain the work of all four areas (viva).

## Phases
1. Hour 0 (all together): agree data contract, feature schema, metric definitions,
   seed and folds, folder layout.
2. Foundations: B ships candidates and features v0 on about 10 trips. C and D build
   against it. A does EDA and the data loader.
3. Experiments: C and D train and compare models with the shared harness. Pick the
   winner by CV macro F1 and log loss. Check calibration before Viterbi.
4. Finalise: D runs Optuna on training trips, refits, evaluates once on student-test.
   B builds Viterbi and tunes on training and student-test. A builds API and UI.
5. Rehearsal: dry run on student-test trips with API on one laptop and UI on another.
   Check API logs. Cross-explain every area.

## Git
- Branches: `feature/<area>-<short-name>`, for example `feature/models-rf`.
- Merge through pull requests. PR template asks: which partition was used, did any
  metric change, does this need a rerun.
- Small, frequent commits under your own name.
- Commit every run JSON and the Optuna study so report numbers are reproducible.
- Never commit secrets, evaluator labels or `api/server_data/`.
