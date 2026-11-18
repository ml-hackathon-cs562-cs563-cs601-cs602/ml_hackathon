PathSnap: Step-by-Step Plan for All Four Members
How to use this plan
Time split: I don’t know your deadline, so each phase is shown as a share of total time. If you have about 3 days, Phase 1 is roughly 15 hours. Adjust the shares to your schedule, but keep the order and the gates.
Gates: Each phase ends with a gate, a short team check (15 minutes) where you confirm the “done when” items before anyone starts the next phase.
Critical path: B’s candidate and feature handoff blocks C and D. A’s API depends on the final artifacts from D and B. Protect those handoffs.
First-phase evaluation: The technical report guide mentions an evaluation before lunch. Confirm what it requires with the organisers, and plan Gate 1 to be ready for it.
Phase	Share of time	Main goal	Gate
0. Kickoff	1 to 2 hours	Contracts, repo, partitions locked	G0
1. Foundations	~20%	Data inventory, candidates v0, features v0, harness, first run	G1
2. Experiments	~30%	Five models compared, Viterbi built, API and UI skeletons	G2 (winner chosen)
3. Tuning and final model	~15%	Optuna, tuned model, decoder tuned, artifacts exported	G3
4. Integration	~15%	End-to-end system on two laptops, all 10 evaluator trips	G4
5. Report and viva prep	~15%	Report assembled, every member can explain every area	G5
6. Submission	Final hours	Public repo, PDF, final rehearsal	Done
Phase 0: Kickoff (all four together, 1 to 2 hours)

Integrator (one person, rotate if you like):

Create the GitHub repository (private until submission). Copy the contents of ml_hackathon/ in, run git init, make the first commit, and push.
Add the team’s GitHub usernames to CODEOWNERS.
Create branches per area: feature/data-a, feature/candidates-b, feature/models-c, feature/models-d, feature/api-ui-a, feature/hmm-b.
Turn .agents/context/workflow_and_ownership.md into a GitHub Project board with one card per task in this plan.

All members together:

Read .agents/AGENTS.md and the context files. These are the rules the coding agent follows, and they are also your rules.
Agree on the data contract in data_contract.md, the candidate table schema, and the metric definitions (macro F1, log loss, observation-level Top-1, candidate miss rate, route continuity).
Agree on the seed and the 5-fold GroupKFold scheme by trip.
Resolve the decisions in open_questions.md and record each answer in docs/decisions/:
Confirm the Decision Tree swap for Gaussian NB.
Decide whether raw data can go in the public repo.
Confirm the UTM zone with a quick check of the coordinates.
Clarify what the first-phase evaluation requires.
Pick the Python version and the UI framework.
Each member installs the environment and confirms it runs.

Member C: generates configs/partitions.yaml with the seed and the trip IDs for the 80 / 10 / 10 split. Commit it. The file is then locked and not changed.

Done when: the repo is pushed, the partitions file is committed, decisions are recorded, and every member has a working environment.

Gate G0: everyone confirms the above. Nobody starts coding before this.

Phase 1: Foundations (~20%)

Goal: the data is understood, the candidate and feature handoff exists, and the shared evaluation harness produces the first real run.

Member A: Data, maps, and EDA
src/pathsnap/ingest/parsers.py: parse .track, .nodes, .arcs, and .route. Check column counts and types. Remember that arc_id is the row index of .arcs, and that GPS rows and node rows are not linked by index.
tests/test_parsers.py: test each parser on one real trip. Do not commit any test data that is not allowed in the repo.
pipelines/01_inspect.py: run on all 100 trips and produce the inventory:
Trip counts per partition, file counts, and observation counts per trip.
Timestamp gaps and the share of steps exactly 1 s apart.
Missing or invalid points, and the geographic extent of each trajectory.
Node and arc counts, connectivity, isolated nodes, and disconnected components.
Trips with unusually large gaps.
Save the output to results/data_inventory.csv and write a data-quality log in docs/decisions/.
src/pathsnap/ingest/data_loader.py: load a partition into memory in one function call. It must not load .route for evaluator trips.
viz/eda/plots.py: coordinate distributions, sampling-interval histogram, missing-value table, and one road-network plot. Save PNGs to report/figures/.
viz/maps/folium_maps.py: one Folium map of a training trip showing its GPS points and its reference route, saved as HTML to report/maps/.
Start report/sections/2_2_eda.md: write the preprocessing answer and the partition used for each analysis.

Done when: the inventory CSV exists, the data-quality log is written, the parsers are tested, and the first map and EDA plots are saved.

Support role: when B is on the critical path, A takes over candidate tests or label tests, or runs the miss-rate check on the 10 trips B uses first.

Member B: Candidates and features (critical path)
geo/projection.py and geo/distances.py: project lon/lat to EPSG:32630 (after confirming the zone). Compute perpendicular distance to an arc in metres, along-arc position, and heading difference. Check lon/lat order.
candidates/generate.py (v0): for each GPS point, return the K nearest arcs within radius R. Output rows keyed by (trip_id, point_idx, candidate_arc_id). Test it on 10 training trips first.
Candidate miss rate: sweep K and R. For each setting, measure how often the true arc is absent from the candidates. Pick the smallest K and R that keep the miss rate low, and record the table in results/.
labels/route_labels.py: label each candidate 1 if its arc is in the trip’s .route, otherwise 0. Add tests/test_labels.py.
features/build.py (v0): perpendicular distance, heading difference, along-arc position, arc length, distance rank, and speed.
pipelines/02_build_candidates.py and 03_build_features.py: run on all 90 training and student trips. Write data/features/features.parquet with features_version: v0.
Announce the handoff in the group chat. This is the most important milestone in Phase 1.
After C’s harness works (later in Phase 1): add neighbour-context features (t-1 and t+1 distances to the same arc) and node degree. Bump to v1, and rerun.

Done when: features.parquet v0 is written and handed off, the candidate miss-rate table is in results/, and labels are tested.

Member C: Shared evaluation harness
utils/seeds.py and utils/timing.py: set all random seeds from one config value. Provide a timer that records training and inference time.
evaluation/metrics.py: implement macro F1, precision, recall, accuracy, log loss, observation-level Top-1 accuracy, candidate miss rate, and route continuity. Test each one on small hand-computed examples in tests/test_metrics.py. Write down your definition of route continuity in docs/decisions/.
evaluation/cv.py: GroupKFold by trip on the 80 training trips, 5 folds from partitions.yaml. Return per-fold scores so stability across folds can be reported.
evaluation/run_logger.py: write one JSON per run to experiments/runs/ using the schema in experiments/README.md.
models/factory.py and models/logreg/model.py: build Logistic Regression from its YAML config. Implement fit, predict_proba, save, and load.
First real run: train Logistic Regression on features v0 through CV and student-test. Commit the run JSON. This proves the whole chain works.
Calibration helper: a reliability curve and log loss check that the team can reuse.

Done when: metrics tests pass, the logistic regression run JSON is committed, and the calibration helper exists.

Member D: Model wrappers and tuning setup
Read the handoff: study the feature schema and the candidate table format.
models/lgbm/model.py: LightGBM wrapper with fit, predict_proba, save, and load. Use configs/models/lgbm.yaml and record the initial hyperparameter values.
models/mlp/model.py: MLP wrapper that saves the training and validation loss history per epoch. This is the only model that gets loss curves.
Dry runs on a small sample of 5 trips using features v0. Do not tune anything yet.
hpo/optuna_search.py skeleton: TPE sampler, GroupKFold on training trips, trial logging. Draft the justified search space in configs/optuna.yaml, writing the reasoning down before any tuning.

Done when: both wrappers run on the sample, the MLP history is saved, and the search space is written with its reasoning.

Gate G1: check the inventory, the candidate miss rate, the features handoff, the harness run, and the wrappers. Confirm the first-phase evaluation content.

Phase 2: Experiments and decoder build (~30%)

Goal: five models are trained and compared on the same folds and metrics, the Viterbi decoder works with any classifier’s probabilities, and the API and UI skeletons work with a stub model.

Member C: Models 1 to 3 and the comparison
Train Logistic Regression (v1 features), Decision Tree, and Random Forest through pipelines/04_train_compare.py. Use the same folds, seed, and metrics for every model.
Record each run’s configuration and timing in its JSON. Do not select on training accuracy.
Check overfitting: compare the CV score with the training score for each model. Record the gap as evidence.
Calibration check for all three models: reliability plot and log loss. Note any model whose probabilities are poorly calibrated, because Viterbi depends on them.
Build results/model_comparison.csv from the run JSONs, not by typing numbers.
Draft report/sections/3_1_model_comparison.md with the table.
Draft the 3.2 subsections for your three models. For each model, answer:
Why it suits PathSnap, and its strengths and limitations as observed in your runs.
Hyperparameters and their initial values, and the loss or objective used.
Student-test performance, Top-1 accuracy, macro F1, and log loss.
Whether you see overfitting or underfitting, and the evidence.

Done when: three runs are logged, the comparison table is generated, and the calibration results are recorded.

Member D: Models 4 and 5
Train LightGBM and the MLP through the same harness with the same folds and seed.
MLP training curves: plot the training and validation loss from the saved history. Do not create curves for any other model.
Calibration check for both models. If their probabilities are poorly calibrated, fit Platt or isotonic calibration on the training folds only, and log the result both with and without calibration.
Draft the 3.2 subsections for LightGBM and the MLP, using the same nine questions.

Done when: both models are logged, the MLP curves are saved, and the calibration results are recorded.

Joint step: Gate G2 (winner selection, 30 to 45 minutes)

The whole team reviews the comparison and agrees on the winner. The decision is based on:

CV macro F1, and its stability across folds.
Log loss and calibration, since Viterbi uses the probabilities.
Student-test results.
Inference speed and model complexity.
Compatibility with the decoder.

Record the decision and the reasoning in docs/decisions/. Do not select the model on training accuracy.

Member B: Viterbi decoder (built on any model’s probabilities)
hmm/emissions.py: convert each candidate’s P(correct) into an emission log-probability. Model a missing candidate explicitly as an absent state, not as a zero probability. Keep the interface general, so it accepts any model.
hmm/transitions.py: build the road network as a graph from the arcs. Score transitions between consecutive candidates:
Same arc: a high score.
Connected arcs: a moderate score.
Otherwise: a penalty based on hop distance versus straight-line distance.
Invalid transitions get a penalty, not an automatic exclusion.
hmm/viterbi.py: implement Viterbi in log space. Document the recurrence in a code comment. Handle missing candidates and GPS gaps, and fall back to the best classifier candidate when a segment is disconnected.
tests/test_viterbi.py: test missing candidates, disconnected arcs, and gap handling on tiny hand-made networks.
Run pipelines/06_decode_hmm.py on student-test trips with the logistic regression probabilities as a placeholder. Produce a first comparison of classifier-only and classifier-plus-HMM.
Draft report/sections/5_hmm_viterbi.md and answer all six questions.

Done when: Viterbi tests pass and the first comparison is produced. After G2, switch the emissions to the winning model’s probabilities.

Member A: API and UI skeletons
api/main.py and api/routes.py:
GET /trips returns the 10 evaluator trip IDs with no labels.
POST /evaluate/{trip_id} runs the pipeline and returns the route. Use a stub that calls the logistic regression model for now.
Log every request to the console, since the evaluator will read these logs.
api/evaluation_service.py: the only module that reads evaluator ground truth. Copy the evaluator route files into api/server_data/ on Laptop 1 only, and check that the folder is gitignored.
ui/app.py: list the trips from the API, call the endpoint for the selected trip, and draw the returned route on a Folium map. A temporary mock response is fine for development, but it must be removed before integration.
viz/maps/folium_maps.py: extend the map so it shows observed GPS points, the reference route, and the predicted route as separate layers, with a legend and layer control.
Drafts: report/sections/6_folium_maps.md and 7_api_ui_deployment.md, with outlines and the first screenshots.

Done when: the API returns a route for a student trip during development, and the UI draws it.

Phase 3: Tuning and final model (~15%)

Goal: the chosen model is tuned, the decoder is tuned, and the artifacts are exported.

Member D: Bayesian optimisation and export
Run Optuna with the TPE sampler on the winning model:
Use the search space in configs/optuna.yaml.
Use GroupKFold on the 80 training trips only.
Run 30 to 50 trials, logging each trial’s parameters, score, and time to hpo/studies/ and results/optuna_trials.csv.
Pick the best configuration, refit on all 80 training trips, and evaluate once on the student-test trips.
Compare untuned and tuned on the same partitions and metrics:
CV macro F1.
Student-test Top-1 accuracy and macro F1.
Log loss.
Inference time.
Report the computational cost of the search.
Do not run more tuning rounds on student-test trips. If you need more trials, go back to CV. The student-test set is for development comparisons, and repeated tuning on it makes the result optimistic.
Export artifacts through pipelines/07_export_artifacts.py: the final model, the fitted preprocessing state, the HMM config, the features version, and the git commit. Write artifacts/README.md with what each file is and how it was produced.
Draft report/sections/4_bayesian_optimisation.md with the six answers.

Done when: the Optuna study is saved, the tuned model’s run JSON is committed, and artifacts are exported with their README.

Member C: Validation and final comparison
Review D’s search space and objective against the harness, to check that the tuning uses only training trips.
Check fold stability for the tuned model, reporting the spread across folds.
Complete results/model_comparison.csv with all five models.
Produce the figures in report/figures/ from the run JSONs:
CV macro F1 comparison across models.
Student-test Top-1 and macro F1 comparison.
Log loss comparison.
Confusion matrix for the winner (optional).
Training and validation loss for the MLP only.
Finish the 3.1 table and your 3.2 subsections.

Done when: every figure is generated by a script from run JSONs, and the table is complete.

Member B: Final decoder tuning and comparison
Plug in the final model’s probabilities and check the emission scores.
Tune the decoder weights (emission weight, transition penalties, gap penalty) on the training and student-test trips only. Log every setting you try.
Produce results/hmm_comparison.csv with pointwise accuracy, sequence-level performance, and route continuity for classifier-only and classifier-plus-HMM.
List the cases where HMM hurt. The guide asks for these explicitly, and including them is more honest.
Write report/sections/5_hmm_viterbi.md in full, with the comparison tables.
Export the HMM config to artifacts/ with D.

Done when: the decoder comparison is final, and the report section is written.

Gate G3: the tuned model, the decoder, and the artifacts exist. Everyone confirms that no evaluator data was touched.

Phase 4: Integration (~15%)

Goal: the full system runs on two laptops, works for all 10 evaluator trips, and can be reproduced from a clean clone.

Member A: API, UI, and the demonstration
Make the API load artifacts once at startup, with the final model, preprocessing, and HMM config.
Replace the stub with the full pipeline: candidates, features, classifier, Viterbi, and then the route.
Remove all UI mocks. The UI must call the API for every result.
Test on the 10 student-test trips first, then on the 10 evaluator trips on Laptop 1, where the server labels are present.
Set up Laptop 2: point PATHSNAP_API_URL at Laptop 1’s address and confirm the UI can reach it on the venue network.
Capture screenshots:
The working UI on Laptop 2 with a predicted route.
The API log in the terminal on Laptop 1 showing successful requests.
These are needed for section 7 of the report.
Update api/README.md and ui/README.md with start commands and the exact steps for the demo.

Done when: all 10 evaluator trips return results through the API, and the screenshots are saved.

Member B: Consistency check
Compare the API output with the offline pipeline output for the same trip. They must match. Fix any difference before continuing.
Search the prediction path for evaluator data:
git grep -n "server_data\|\.route" -- src pipelines ui

Only api/evaluation_service.py may reference evaluator route files.
3. Run the Viterbi tests again on the final code.

Done when: the offline and API outputs match, and the search finds no evaluator references outside api/.

Member C: Reproducibility check
Clone the repo into a fresh folder and follow the README exactly.
Rerun pipelines/01 through 07 in order.
Compare the new run JSONs and the tables in results/ with the committed ones. They must match, given the fixed seeds.
Check that every number in the report sections appears in a file under results/ or experiments/runs/.

Done when: the rerun reproduces the committed results.

Member D: Artifacts and environment
Load the artifacts in a fresh environment and check that predictions match.
Pin requirements.txt with exact versions, and test the install on a second machine.
Check model file sizes. If the weights are large, confirm that Git LFS or the release asset works, and write the download step in the README.

Done when: the artifacts load in a clean environment, and the requirements install.

Joint step: Gate G4 (dry run, about 1 hour)
Run the full demo on two laptops, as the evaluator will see it.
Select each of the 10 evaluator trips in the UI and confirm a result.
Check the API logs for every request.
Freeze the code and tag it v1.0-rc.
Phase 5: Report and viva preparation (~15%)

Goal: one consolidated PDF that matches the repo, and every member can explain the whole system.

Report ownership and assembly
Section	Owner
2.2 Dataset understanding and EDA	A
3.1 Comparison table and 3.3 Visualisations	C
3.2 Per-model subsections	C (LogReg, DTree, RF) and D (LightGBM, MLP)
4 Bayesian optimisation	D
5 HMM and Viterbi	B
6 Folium maps	A
7 API, UI, and deployment	A

Integrator: assembles the PDF from report/sections/ and includes the AI-assistance disclosure that the event requires.

Each owner finishes their sections in the next 2 to 3 hours, then:

Checks every number against results/ or experiments/runs/.
Checks that every figure is generated by a script.
Checks that every section states which partition was used for each analysis.

Final PDF checks:

At least one Folium map as a screenshot.
UI and API screenshots from the receiving laptop.
No training curves for models without epoch histories.
No invented citations, metrics, or results.
Viva preparation

Cross-explanation sessions (30 minutes per area). Each owner presents their area, and the other three ask questions.

Questions every member should be able to answer:

Why was this model suitable, and why was the winner chosen?
Why was log loss used alongside macro F1?
How do you prevent leakage between partitions?
How do the candidates work, and what is the candidate miss rate?
How does Viterbi handle a missing candidate or a disconnected transition?
What happens when a GPS observation has no good candidate?
Why was Bayesian optimisation done only on training trips?
Which projection did you use, and why?

Each member prepares a one-page cheat sheet for their own area.

Gate G5: each member explains one area they did not build, without notes.

Phase 6: Submission

Repository:

Make the repo public.
Check that the README installation and run steps work when followed by someone who did not write them.
Check for secrets:
git grep -in "password\|api_key\|secret\|token"
Confirm that api/server_data/ and .env are not committed.
Confirm that the Git LFS files or the release asset download works.
Tag the final release, for example v1.0.

Report: submit the single consolidated PDF.

Final rehearsal: run the demo on two laptops once more, and confirm all 10 evaluator trips.

Summary by member
Member	Phase 1	Phase 2	Phase 3	Phase 4	Phase 5
A	Parsers, inspection, data loader, EDA, Folium maps	API and UI skeletons, map layers	Support API and UI integration	API, UI, all 10 trips, screenshots	Sections 2.2, 6, 7
B	Candidates, miss rate, labels, features v0, then v1	Viterbi, transitions, tests, first comparison	Final decoder tuning, comparison, artifacts	Offline vs API consistency check	Section 5
C	Metrics, CV, run logger, seeds, first logistic regression run	LogReg, DTree, RF, calibration, comparison table	Validate tuning, figures, fold stability	Reproducibility check	Sections 3.1, 3.3, and 3.2 for three models
D	LightGBM and MLP wrappers, search space	LightGBM and MLP runs, MLP curves, calibration	Optuna, tuned model, export	Artifacts in a clean environment, pinned requirements	Section 4 and 3.2 for two models