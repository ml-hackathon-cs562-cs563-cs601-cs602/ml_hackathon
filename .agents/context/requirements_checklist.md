# Requirements checklist (from the three supporting documents and the hackathon guide)

## Models and experimentation
- [ ] At least five distinct models, from distinct algorithm families, trained and evaluated.
- [ ] Each model's configuration recorded. Training and inference time recorded.
- [ ] Model choice justified by experimental evidence. Not by training accuracy alone.
- [ ] Use the 80 training trips for fitting and CV. Use the 10 student-test trips for comparisons.
- [ ] Preprocessing applied consistently and fitted on training data only.
- [ ] Logs, random seeds, configs and results retained.

## Metrics
Report where applicable: accuracy, precision, recall, macro F1, confusion matrix, log loss,
observation-level Top-1 accuracy, candidate miss rate, route continuity, sequence-level
performance, training time, inference time. State the primary selection metric and why.

## Bayesian hyperparameter optimisation (chosen model only)
- [ ] Justified search space. Stated objective. Group-aware CV on training trips.
- [ ] Every trial recorded. Best configuration identified. Refit. One evaluation on student-test trips.
- [ ] Tuned vs untuned comparison on the same partitions and metrics.
- [ ] Computational cost reported.
- [ ] Evaluator partition not used to choose search space, objective, hyperparameters, model or thresholds.

## HMM and Viterbi
- [ ] Hidden-state representation described.
- [ ] Emission and transition modelling choices explained. Viterbi recurrence explained.
- [ ] Missing candidates, invalid transitions and GPS gaps handled.
- [ ] A predicted arc returned for each relevant GPS observation.
- [ ] Classifier-only vs classifier-plus-HMM compared (pointwise accuracy and sequence-level).
- [ ] Route continuity analysed. Decoder parameters tuned on training and student-test data only.

## API, UI and deployment
- [ ] FastAPI backend processes evaluation requests and loads artifacts once at startup.
- [ ] UI communicates with the API and displays the predicted route and results.
- [ ] Ground truth stays server-side. No hardcoded outputs. API logs visible on Laptop 1.
- [ ] Works for all ten evaluator trips.
- [ ] Exported artifacts: final model, preprocessing, HMM config, other inference assets.

## Repository (must be public)
- [ ] Data processing, feature engineering, model training, evaluation, hyperparameter
      optimisation, HMM/Viterbi, API, UI, deployment config, visualisation code, supporting results.
- [ ] Saved model weights and configs (subject to size limits). Code that produced them too.
- [ ] README with installation, execution and deployment instructions.
- [ ] Clearly named folders. No passwords, API keys or secrets.

## Technical report (one consolidated PDF)
- [ ] 2.2 Dataset understanding and EDA (preprocessing, patterns, features, missing values,
      trajectory plots, road-network visuals, Folium map, partition named for each analysis).
- [ ] 3.1 Consolidated comparison table. 3.2 One subsection per model (answer all 9 questions).
- [ ] 3.3 Visualisations: CV macro F1, student Top-1 and macro F1, log loss, confusion matrices (optional).
      Training curves only for models with epoch histories.
- [ ] 4 Bayesian optimisation answers (6 questions) and tuned vs untuned comparison.
- [ ] 5 HMM and Viterbi answers (6 questions) and classifier-only vs HMM comparison.
- [ ] 6 Folium maps: GPS observations, road network, reference route, predicted route,
      legend, layer control. At least one screenshot in the PDF. Map code committed.
- [ ] 7 API and UI: UI screenshots and API response screenshots from the receiving laptop's terminal.
- [ ] Every number reproducible from the repo.
- [ ] AI assistance disclosed as the event policy requires.

## Phase timing
The technical report guide notes a first-phase evaluation before lunch. Confirm its
requirements with organisers and plan a demonstrable milestone before then.
