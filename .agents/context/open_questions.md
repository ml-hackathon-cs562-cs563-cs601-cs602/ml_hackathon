# Open questions (resolve before the matching work starts)

Each item is a team or organiser decision. Do not guess. Record the answer in `docs/decisions/`.

1. Model families: the guide's table lists Logistic Regression and Gaussian Naive Bayes
   together under "linear and probabilistic". The plan replaces Gaussian NB with a Decision
   Tree so the five models come from five families. Confirm with the team.
2. Raw data policy: may the raw dataset be committed to the public repository? If not,
   `data/raw/` stays out of Git and the README documents the download step.
3. CRS: confirm the UTM zone against the actual coordinates in `01_inspect.py`.
4. Candidate parameters: K and R, set from the candidate miss rate measurement.
5. Model weights: Git LFS or a release asset? Size limits to check.
6. First-phase evaluation (before lunch): what exactly is required?
7. Python version and UI framework (Streamlit or another option).
8. Folder layout: this repo uses `ingest/` (not `io/`) to avoid shadowing Python's standard `io` module.
