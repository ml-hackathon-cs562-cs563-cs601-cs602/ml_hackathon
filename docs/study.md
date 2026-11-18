# PathSnap: Study Document

**Last Updated:** November 2026  
**Team:** 4-member ML hackathon team  
**Repository:** Public GitHub repo (to be released)

---

## 1. Problem Statement

### 1.1 What is the Problem?

**GPS Map-Matching in Urban Environments**

GPS tracking in urban areas suffers from noise caused by multipath effects, signal blockage from buildings, and low sampling rates. A vehicle can appear to jump onto a parallel street or across buildings, making it difficult to determine its actual path through a road network.

**PathSnap** is a distributed ML system that:
1. Matches noisy GPS observations to a road network
2. Reconstructs realistic trajectories that respect road connectivity
3. Outputs a sequence of road segments (arcs) that the vehicle likely traveled

### 1.2 Deployment Architecture

The system runs on two laptops:

| Laptop | Components | Purpose |
|--------|-----------|---------|
| Laptop 1 | Trained model, HMM/Viterbi, road network, FastAPI server, evaluator ground truth | Runs the API and has access to labels |
| Laptop 2 | Evaluator UI | Lists evaluator trips, sends trip IDs to API, displays predicted routes |

**Constraint:** The UI must work for all 10 evaluator trips, not just one demo trip. Ground truth stays server-side—no hardcoded outputs or bypassing the API.

### 1.3 Dataset Overview

| Partition | Trips | Purpose |
|-----------|-------|---------|
| Training | 80 | Model fitting, cross-validation, Bayesian optimization |
| Student-test | 10 | Development comparisons, decoder tuning |
| Evaluator-test | 10 | Final evaluation only (labels never visible during development) |

**Data files per trip:**
- `.track` — GPS observations (lon, lat, timestamp_s), nominally 1 Hz
- `.nodes` — Road network nodes (lon, lat), row index = node index
- `.arcs` — Road segments (from_node_id, to_node_id), row index = arc_id
- `.route` — Reference route (ordered arc_ids), provided for train and student trips only

**Typical trip statistics (to be verified):**
- ~2,600 GPS points per trip (up to ~19,000)
- Node counts: 300 to 155,000
- Arc counts: 650 to 307,000
- Coordinates around lon -1.86, lat 52.96 (UTM zone 30N)

---

## 2. Our Approach

### 2.1 High-Level Architecture

PathSnap uses a **two-stage pipeline**:

```
GPS observations → Candidate generation → Feature extraction → Classifier → HMM/Viterbi → Predicted route
```

**Stage 1: Candidate Scoring (Supervised ML)**
- For each GPS point, generate K candidate arcs within radius R
- Extract features for each (point, candidate) pair
- Train a binary classifier to predict P(candidate is correct)
- Output: probability scores for all candidates

**Stage 2: Sequence Decoding (HMM/Viterbi)**
- States: candidate arcs for each GPS point
- Emissions: classifier probabilities (converted to log-likelihoods)
- Transitions: based on road connectivity and geometric consistency
- Viterbi algorithm finds the most likely sequence of arcs

### 2.2 Why This Approach?

| Aspect | Rationale |
|--------|-----------|
| **Candidate-based** | Each trip has its own road network, so arc IDs are trip-specific. A candidate-based approach allows training a single binary classifier across all trips. |
| **Binary classification** | Simpler than multi-class; handles variable numbers of candidates per point; probability outputs feed naturally into HMM. |
| **HMM for sequence** | Exploits temporal continuity—vehicles don't teleport. Reduces zigzagging between arcs. |
| **GroupKFold CV** | Prevents leakage: all observations from the same trip stay in the same fold. |

### 2.3 Data Flow

```
raw_data.zip + pathsnap_bundle.zip
         ↓
    [Unpack & audit]
         ↓
    [Candidate generation] ← K=20, R=50m (to be swept)
         ↓
    [Feature extraction] ← 15 features per (point, candidate)
         ↓
    [features.parquet] ← Handoff from B to C and D
         ↓
    [Model training] ← 5 models, 5-fold CV on training trips
         ↓
    [HMM decoding] ← Viterbi on student-test trips
         ↓
    [Final route output]
```

---

## 3. Feature Engineering

### 3.1 Features Selected (15 features)

Features are defined in `src/pathsnap/features/build.py` and versioned by `features_version`.

| # | Feature | Type | Description | Rationale |
|---|---------|------|-------------|-----------|
| 1 | `dist` | Geometric | Perpendicular distance from GPS point to candidate arc (meters) | Strongest signal—closer arcs more likely correct |
| 2 | `rank` | Ranking | Rank by distance among K candidates (0-indexed) | Encodes relative position—closest often correct |
| 3 | `dist_rel` | Geometric | Relative distance: `dist - dist_min` for this GPS point | Highlights advantage over other candidates |
| 4 | `dist_gap` | Geometric | Distance gap to next-ranked candidate (meters) | Large gaps indicate clear winner |
| 5 | `t_proj` | Geometric | Projected position along the arc (normalized) | Helps identify alignment with arc geometry |
| 6 | `arc_len` | Arc property | Length of candidate arc (meters) | Longer arcs may be major roads (higher prior) |
| 7 | `bear_diff_cos` | Geometric | Cosine of bearing difference (weighted by speed) | Captures heading alignment; cosine smooths angular wraparound |
| 8 | `bear_diff_abs` | Geometric | Absolute bearing difference between GPS trajectory and arc (radians, weighted by speed) | Large differences indicate mismatch |
| 9 | `speed` | Motion | Estimated speed at GPS observation (m/s) | Speed constraints can rule out impossible matches |
| 10 | `n_cand` | Candidate set | Number of candidates for this GPS point | Sparse candidates may indicate GPS dropout or off-road |
| 11 | `dist_x_cos` | Geometric | Interaction: `dist * cos(bearing_diff)` | Combines distance and heading into single feature |
| 12 | `perp_signed` | Geometric | Signed perpendicular distance (left/right of arc) | Captures which side of the road; useful for multi-lane roads |
| 13 | `prev_conn` | Connectivity | 1 if candidate arc connects to any candidate at previous GPS point (t-1) | Temporal continuity—connected arcs more likely |
| 14 | `next_conn` | Connectivity | 1 if candidate arc connects to any candidate at next GPS point (t+1) | Temporal continuity—connected arcs more likely |
| 15 | `dt` | Temporal | Time difference from previous GPS observation (seconds, capped at 60s) | Time gaps indicate GPS dropout; affects motion estimation |

**Feature set versioning:**
- `v0_pipelineB_15feat`: Initial feature set (15 features from Member B's implementation)
- Future versions require team decision and new version label

### 3.2 Features Excluded and Why

| Excluded Feature | Reason |
|------------------|--------|
| **Trip ID** | Categorical, trip-specific; would cause overfitting to training trips |
| **Partition** | Metadata, not predictive; would leak partition information |
| **Point index** | Ordinal position within trip; not predictive of correct arc |
| **Candidate arc ID** | Trip-specific; no generalization across trips |
| **Label** | Target variable; cannot be used as a feature |
| **Source node / Destination node** | Trip-specific IDs; included in candidate table for C's metrics but excluded from model features |
| **Absolute coordinates (lon/lat)** | Each trip has different geographic extent; would not generalize across trips |
| **Road type / Speed limit** | Not available in the dataset (no road metadata provided) |
| **Historical traffic patterns** | Not available in the dataset |
| **Node degree** | While mentioned in docs/feature_engineering.md as potential v1 feature, not implemented in current FEATS list |
| **Raw timestamp** | Not informative; only relative time differences (`dt`) matter for motion estimation |

### 3.3 Preprocessing Pipeline

1. **Coordinate projection:** lon/lat (EPSG:4326) → UTM (EPSG:32630) for metric distances
2. **Missing value handling:** Features should not have missing values; impute with median if needed
3. **Feature scaling (MLP only):** StandardScaler fitted on training data, applied to validation/test
4. **No feature selection:** Initial feature set kept; removal requires team decision

### 3.4 Feature Quality Checks

Before training:
- ✓ No NaNs in features
- ✓ No inf values
- ✓ Finite values only
- ✓ Distributions checked for unexpected outliers
- ✓ Correlation matrix computed (pairs with |r| > 0.95 flagged)

---

## 4. Models

### 4.1 Five Models Trained

| # | Model | Family | Owner | Key Characteristics |
|---|-------|--------|-------|----------------------|
| 1 | Logistic Regression | Linear | Member C | Baseline; interpretable; calibrated probabilities |
| 2 | Decision Tree | Tree | Member C | Interpretable; handles non-linear relationships; prone to overfitting |
| 3 | Random Forest | Ensemble (bagging) | Member C | Reduces variance; feature importance; robust |
| 4 | LightGBM | Ensemble (boosting) | Member D | Leaf-wise growth; high accuracy; fast inference |
| 5 | MLP | Neural network | Member D | 2 hidden layers (64, 32), ReLU, Adam; requires scaling; **only model with epoch-wise loss history** |

### 4.2 Training Configuration

| Aspect | Setting |
|--------|---------|
| **Cross-validation** | 5-fold GroupKFold by trip (80 training trips) |
| **Seed** | 42 (from `configs/partitions.yaml`) |
| **Early stopping** | Monitor validation log loss on held-out training trips (15% inner split) |
| **Calibration** | Sigmoid/isotonic regression via trip-grouped CV; kept only if log loss improves |
| **Metrics** | Macro F1, log loss, Top-1 accuracy, precision, recall, route continuity |

### 4.3 Model Comparison (After Training)

Selection criteria:
1. CV macro F1 and stability across folds
2. Log loss and calibration quality (critical for Viterbi)
3. Student-test Top-1 accuracy and macro F1
4. Inference speed (API latency matters)
5. Model complexity and interpretability

**Winner selection at Gate G2** (after Phase 2 experiments).

---

## 5. HMM/Viterbi Decoder

### 5.1 Hidden State Representation

- **States:** Candidate arcs for each GPS point (up to K states per observation)
- **Observations:** GPS coordinates (lon, lat)
- **Goal:** Find the most likely sequence of arcs

### 5.2 Emission Probabilities

- Derived from classifier output: P(candidate is correct | features)
- Converted to log-likelihood: log(P) + constant
- Missing candidates modeled explicitly (not as zero probability)

### 5.3 Transition Probabilities

| Transition Type | Score | Rationale |
|-----------------|-------|-----------|
| Same arc | High | Vehicle stays on same road segment |
| Connected arcs | Moderate | Valid turn or continuation |
| Disconnected arcs | Penalty based on hop distance vs straight-line distance | Geometric consistency |

### 5.4 Viterbi Implementation

- Log-space computation to avoid underflow
- Handles missing candidates and GPS gaps
- Falls back to best classifier candidate when disconnected

---

## 6. Bayesian Optimization

### 6.1 Scope

- **Target:** Winning model only (chosen at Gate G2)
- **Search space:** Defined in `configs/optuna.yaml` with justification
- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **Objective:** CV macro F1 or log loss (to be decided)
- **Trials:** 30-50

### 6.2 Constraints

- Optimization uses **only** the 80 training trips
- GroupKFold by trip (same as model training)
- Student-test trips used for **one** final comparison (tuned vs untuned)
- **No** tuning on evaluator data

### 6.3 Outputs

- Best hyperparameter configuration
- Comparison table: untuned vs tuned (CV, student-test, inference time)
- Computational cost reported
- Run JSON with all trial logs

---

## 7. Next Phase (Phase 2: Experiments and Decoder Build)

### 7.1 Immediate Tasks

| Member | Task | Dependencies |
|--------|------|--------------|
| **Member C** | Train Models 1-3 (LogReg, Decision Tree, Random Forest) | Features v0 from B |
| **Member D** | Train Models 4-5 (LightGBM, MLP) | Features v0 from B |
| **Member B** | Build Viterbi decoder (emissions, transitions, tests) | Any model's probabilities (placeholder) |
| **Member A** | API and UI skeletons, Folium map extensions | None |

### 7.2 Gate G1 Checklist (Before Starting Phase 2)

- [ ] Data inventory CSV exists (`results/data_inventory.csv`)
- [ ] Candidate miss rate table exists (swept K and R)
- [ ] Features v0 handoff complete (`data/features/features.parquet`)
- [ ] First logistic regression run JSON committed
- [ ] Calibration helper exists
- [ ] LightGBM and MLP wrappers run on sample
- [ ] Optuna search space drafted with reasoning

### 7.3 Gate G2: Winner Selection (End of Phase 2)

After training all 5 models:
1. Build comparison table from run JSONs (no manual entry)
2. Review CV macro F1 stability, log loss, calibration, student-test results
3. **Team decision:** Choose winner based on experimental evidence
4. Record decision and reasoning in `docs/decisions/`

### 7.4 Phase 3 Preview (Tuning)

- Bayesian optimization on winning model
- Decoder tuning (transition weights, penalties)
- Artifact export (model, scaler, HMM config, feature version)
- Reproducibility checks

---

## 8. Technical Constraints and Rules

### 8.1 Data Safety

| Rule | Enforcement |
|------|-------------|
| Evaluator trips never enter training/feature selection/hyperparameter choice | Partition file locked; API-only access to evaluator labels |
| Ground truth only in `api/server_data/` | Only `api/evaluation_service.py` reads it |
| No hardcoded predictions | API must process every request |
| Partitions from `configs/partitions.yaml` only | File committed and locked |
| Fit preprocessing on training trips only | StandardScaler fitted on training data |

### 8.2 Reproducibility

- Seeds from `configs/partitions.yaml` (seed: 42)
- Every training run writes JSON to `experiments/runs/`
- Report tables generated from run JSONs (no manual entry)
- Metrics come from `src/pathsnap/evaluation/metrics.py`

### 8.3 Coordinate System

- Input: lon/lat (WGS84, EPSG:4326), **lon first**
- Output: UTM (EPSG:32630), zone 30N (to be verified)
- All distance calculations in meters after projection

---

## 9. Member Responsibilities

| Member | Primary Areas |
|--------|---------------|
| **Member A** | Data parsing, EDA, Folium maps, API, UI, deployment |
| **Member B** | Candidate generation, features, HMM/Viterbi, decoder tuning |
| **Member C** | Metrics, CV harness, Models 1-3, comparison table, validation |
| **Member D** | Models 4-5 (LightGBM, MLP), Bayesian optimization, artifact export |

---

## 10. File Locations

| Artifact | Path |
|----------|------|
| Raw data | `data/raw/train/`, `data/raw/student_test/` |
| Features | `data/features/features.parquet` |
| Models | `src/pathsnap/models/*/model.py` |
| Training runs | `experiments/runs/*.json` |
| Results | `results/*.csv` |
| Artifacts | `artifacts/` |
| API | `api/main.py`, `api/routes.py` |
| UI | `ui/app.py` |
| Configs | `configs/*.yaml` |
| Documentation | `docs/*.md`, `.agents/context/*.md` |

---

## 11. Open Questions

(From `.agents/context/open_questions.md`)

| Question | Status | Decision |
|----------|--------|----------|
| Decision Tree swap for Gaussian NB | Resolved | Decision Tree chosen |
| Raw data in public repo | TBD | Team decision needed |
| UTM zone confirmation | TBD | Verify with coordinate check (lon -1.86, lat 52.96) |
| First-phase evaluation requirements | TBD | Confirm with organizers |
| Python version | TBD | Team decision (3.10+ recommended) |
| UI framework | TBD | Team decision |

---

## 12. References

1. **Newson & Krumm (2009):** "Hidden Markov Map Matching Through Noise and Sparseness" — foundational HMM approach
2. **Quddus et al. (2007):** "Current map-matching algorithms for transport applications" — comprehensive review
3. **LightGBM Documentation:** https://lightgbm.readthedocs.io/
4. **Optuna Documentation:** https://optuna.readthedocs.io/

---

**End of Study Document**
