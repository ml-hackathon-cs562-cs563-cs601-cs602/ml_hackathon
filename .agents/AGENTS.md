# AGENTS.md: instructions for coding agents

Read this file first, then the files in `.agents/context/`.

## What this project is
PathSnap is a machine learning hackathon project. It matches noisy GPS observations
to a road network and reconstructs a realistic route. A four-person team builds it in
this monorepo. The work is graded on correctness, experimental rigour, reproducibility,
and whether every member can explain the work.

## Your role
You help write code for the file or folder the human asks about. Stay inside that
owner's area (see `.agents/context/workflow_and_ownership.md`). If a change needs
another area, say so and stop. Do not edit other owners' files without being asked.

Many files in this repo are stubs that contain only a comment describing what the file
must do. Read that comment, then implement exactly that, following the rules below.

## Non-negotiable rules
1. Evaluator trips (10) never enter training, feature selection, threshold selection,
   hyperparameter choice, or the prediction pipeline or UI.
2. Evaluator ground truth lives only in `api/server_data/`. Only
   `api/evaluation_service.py` reads it. Never import it into `src/pathsnap/` or `pipelines/`.
3. No hardcoded predictions, no lookup of expected answers, no bypassing the API.
   The UI must go through the API for every result.
4. Partitions come only from `configs/partitions.yaml`. No trip is split across partitions.
5. Fit every preprocessing step on training trips only. Apply the fitted state to other trips.
6. Use only the 80 training trips for model fitting, CV and Bayesian optimisation.
   Student-test trips are for development comparisons and decoder tuning.
7. Every training run writes one JSON to `experiments/runs/` (schema in
   `experiments/README.md`). Report tables in `results/` are generated from these files.
   Never type a metric into a report by hand.
8. Never invent numbers, results, citations, paper titles or URLs. If a value is unknown,
   write `TODO(team): ...` and add it to `.agents/context/open_questions.md`.
9. Lon/lat are angular. Project to the metric CRS in `configs/data.yaml` before any
   distance calculation. Verify the UTM zone against the data.
10. Only models with epoch-wise histories (the MLP) get training and validation loss curves.
    Do not create curves for other models.
11. Do not copy code, configs or results from other teams or from unattributed sources.
    Write code for this project. Cite any external library or snippet you adapt.
12. No secrets, passwords or API keys in code, configs or commits. Use `.env` (gitignored).
13. Read paths and parameters from `configs/`. Do not hardcode them in code.

## Pipeline (candidate scorer + Viterbi)
1. Project coordinates to the metric CRS.
2. For each GPS point, take K nearest arcs within radius R (candidates).
3. Build features per (trip, point, candidate). Label 1 if the arc is in the trip's `.route`.
4. Classifier outputs P(candidate is correct) per row.
5. Viterbi over candidate states: emissions from classifier probabilities, transitions from
   road connectivity and hop distance versus straight-line distance.
6. Output one arc per relevant GPS observation, then the predicted route.

Each trip has its own road network, so `arc_id` means something different in every trip.
Do not train a single multi-class model over arc IDs.

## Coding conventions
- Python 3.10+ (confirm with team). Type hints, `pathlib`, docstrings on public functions.
- Keep logic in `src/pathsnap/` as pure functions. `pipelines/` scripts are thin wrappers.
- Seeds come from `configs/partitions.yaml`. Every random source must be seeded.
- Log training time and inference time for every model.
- Add or update tests in `tests/` for any parser, label, metric or decoder change.
- Keep code readable for the viva: clear names, short functions, comments on non-obvious
  decisions (for example why a transition penalty was chosen).

## Before you finish a task
- Run the relevant test(s) in `tests/`.
- Confirm no evaluator file or label is referenced outside `api/`.
- Confirm any new number in a report comes from a run JSON.
- Say what you changed, what you did not verify, and any TODO you added.
