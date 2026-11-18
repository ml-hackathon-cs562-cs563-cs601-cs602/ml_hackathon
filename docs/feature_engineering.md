# Feature Engineering for PathSnap

**Owner:** Member B (feature implementation), Member D (documentation)  
**Last Updated:** November 2026  
**Version:** v0 (initial), v1 (with neighbour context)

---

## Overview

PathSnap uses a candidate-based approach for map-matching. For each GPS observation point, we generate K candidate arcs within radius R. Each candidate is scored by a binary classifier that predicts the probability that the candidate arc is part of the true route.

This document describes the features engineered for each (trip, point, candidate) tuple.

---

## Feature Categories

### 1. Geometric Features

These features capture the geometric relationship between the GPS point and the candidate arc.

#### 1.1 Perpendicular Distance (`dist_m`)

- **Description:** The perpendicular distance from the GPS point to the candidate arc, in meters.
- **Calculation:** Project the GPS point onto the line segment defined by the arc, compute Euclidean distance.
- **Units:** Meters
- **CRS:** Projected to EPSG:32630 (UTM zone 30N) before calculation.
- **Typical Range:** 0-50 meters (depends on candidate radius R)
- **Importance:** High - closer arcs are more likely to be correct.

#### 1.2 Along-Arc Position (`along_arc_position`)

- **Description:** The normalized position of the projection point along the arc.
- **Calculation:** Distance from arc start to projection point / total arc length.
- **Range:** [0, 1]
- **Interpretation:** 0 = at arc start, 1 = at arc end, 0.5 = midpoint.
- **Importance:** Medium - helps identify if GPS point aligns with arc geometry.

#### 1.3 Heading Difference (`heading_diff`)

- **Description:** The angular difference between the GPS trajectory heading and the arc bearing.
- **Calculation:** 
  - GPS heading: bearing between consecutive GPS points (t-1 to t)
  - Arc bearing: bearing from arc start to arc end
  - Difference: absolute angular difference, accounting for wraparound
- **Units:** Degrees
- **Range:** [0°, 180°]
- **Importance:** High - large differences indicate mismatch.

### 2. Arc Features

Features derived from the candidate arc's properties.

#### 2.1 Arc Length (`arc_length`)

- **Description:** The length of the candidate arc in meters.
- **Calculation:** Euclidean distance between arc endpoints (projected coordinates).
- **Units:** Meters
- **Importance:** Medium - longer arcs may be major roads (higher prior probability).

#### 2.2 Node Degree (`node_degree`)

- **Description:** The connectivity of the arc's end nodes.
- **Calculation:** Number of arcs connected to the arc's end node.
- **Range:** 1 (dead-end) to N (intersection)
- **Importance:** Medium - high connectivity may indicate intersections (turns).

### 3. Ranking Features

Features that encode relative position among candidates.

#### 3.1 Distance Rank (`distance_rank`)

- **Description:** The rank of this candidate by perpendicular distance among all K candidates for this GPS point.
- **Calculation:** Sort candidates by `dist_m`, assign rank (1 = closest).
- **Range:** [1, K]
- **Importance:** High - top-ranked candidates are more likely correct.

### 4. Motion Features

Features derived from GPS trajectory dynamics.

#### 4.1 Speed (`speed`)

- **Description:** Estimated speed at this GPS observation.
- **Calculation:** Distance between GPS points (t-1 to t) / time difference.
- **Units:** meters/second
- **Importance:** Medium - speed constraints can rule out impossible matches.
- **Caveat:** GPS noise can cause speed errors, especially in urban canyons.

### 5. Neighbour Context Features (v1)

Features that consider the relationship between consecutive GPS observations and their candidates.

#### 5.1 Previous Point Distance (`prev_dist_to_same_arc`)

- **Description:** Distance from the previous GPS point (t-1) to this candidate arc.
- **Calculation:** Same as `dist_m`, but for point t-1.
- **Units:** Meters
- **Importance:** High - continuity suggests the same arc.

#### 5.2 Next Point Distance (`next_dist_to_same_arc`)

- **Description:** Distance from the next GPS point (t+1) to this candidate arc.
- **Calculation:** Same as `dist_m`, but for point t+1.
- **Units:** Meters
- **Importance:** High - continuity suggests the same arc.

---

## Feature Selection

### Initial Feature Set (v0)

The initial model uses these features:
1. `dist_m` - Perpendicular distance
2. `heading_diff` - Heading difference
3. `along_arc_position` - Position along arc
4. `arc_length` - Arc length
5. `distance_rank` - Distance rank
6. `speed` - Estimated speed

**Rationale:** These features capture the essential geometric and dynamic information needed to score candidates.

### Extended Feature Set (v1)

Additional features:
7. `prev_dist_to_same_arc` - Previous point distance
8. `next_dist_to_same_arc` - Next point distance
9. `node_degree` - Node connectivity

**Rationale:** Context features improve temporal consistency and reduce zigzagging between arcs.

### Feature Selection Techniques

1. **Correlation Analysis:** Remove highly correlated features (r > 0.95).
2. **Feature Importance:** Use LightGBM feature importance to identify most predictive features.
3. **Recursive Feature Elimination:** Optionally, use RFE to find optimal subset.
4. **Domain Knowledge:** Retain features with clear physical interpretation.

---

## Preprocessing Pipeline

### 1. Coordinate Projection

All distance calculations require metric coordinates:
- **Input:** lon/lat (WGS84, EPSG:4326)
- **Output:** UTM coordinates (EPSG:32630)
- **Tool:** `pyproj.Transformer`

```python
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
x, y = transformer.transform(lon, lat)
```

### 2. Feature Scaling

For models that require scaling (MLP, Logistic Regression):
- **Method:** StandardScaler (zero mean, unit variance)
- **Fit:** On training data only
- **Apply:** Transform validation and test data using training scaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Use training scaler!
```

**Important:** Never fit the scaler on validation or test data. This causes data leakage.

### 3. Missing Value Handling

Current features should not have missing values. If they occur:
- **Distance features:** Should never be missing (candidates always have positions)
- **Heading difference:** May be missing for first/last points. Impute with median.
- **Speed:** May be missing for first point. Impute with median.

### 4. Categorical Encoding

No categorical features in current set. If added (e.g., road type):
- **Method:** One-hot encoding for low-cardinality, target encoding for high-cardinality.

---

## Data Schema

### Candidate Table (features.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `trip_id` | string | Trip identifier |
| `partition` | string | 'train' or 'student' |
| `point_idx` | int | GPS observation index (0-based) |
| `candidate_arc_id` | int | Candidate arc index (0-based, trip-specific) |
| `label` | int | 1 if arc is in true route, else 0 |
| `dist_m` | float | Perpendicular distance (meters) |
| `heading_diff` | float | Heading difference (degrees) |
| `along_arc_position` | float | Position along arc [0, 1] |
| `arc_length` | float | Arc length (meters) |
| `distance_rank` | int | Rank by distance [1, K] |
| `speed` | float | Estimated speed (m/s) |
| `prev_dist_to_same_arc` | float | Distance from t-1 to this arc (v1) |
| `next_dist_to_same_arc` | float | Distance from t+1 to this arc (v1) |
| `node_degree` | int | Connectivity of end node (v1) |

---

## Feature Engineering Rationale

### Why These Features?

1. **Distance (`dist_m`):** The strongest signal for correct match. GPS points near the true arc should have lower distance.

2. **Heading (`heading_diff`):** GPS trajectories follow road geometry. Large heading differences indicate unlikely matches.

3. **Rank (`distance_rank`):** Encodes relative position. Even if absolute distance is uncertain, the closest candidate is often correct.

4. **Context (`prev_dist_to_same_arc`, `next_dist_to_same_arc`):** Vehicles don't teleport. Consecutive points should match nearby arcs.

5. **Arc Properties (`arc_length`, `node_degree`):** Road network structure provides prior probabilities.

### What We Avoided

1. **Future Information:** Features only use past and current observations, never future points (except v1 context features, which use t+1 for smoothing).

2. **Labels:** Features are computed without using the true route. Only labels use ground truth.

3. **Evaluator Data:** No features are computed from evaluator trips during development.

---

## Computational Considerations

### Performance

- **Feature Computation:** O(N × K) where N = number of GPS points, K = candidates per point
- **Typical Dataset:** 80 trips × 2600 points × 20 candidates ≈ 4M rows
- **Memory:** Features DataFrame ~500MB for 4M rows with 10 float columns

### Optimization Tips

1. **Vectorization:** Use NumPy/Pandas vectorized operations, avoid loops.
2. **Projection Cache:** Project all GPS points once at the start.
3. **Spatial Indexing:** Use KD-tree or Ball-tree for candidate generation.
4. **Parallelization:** Process trips in parallel (each trip is independent).

---

## Validation

### Feature Quality Checks

1. **Distribution:** Plot histograms for each feature. Check for unexpected modes or outliers.
2. **Correlation Matrix:** Identify highly correlated features for potential removal.
3. **Feature vs. Label:** Plot feature distributions for label=0 vs label=1.
4. **Temporal Stability:** Check if feature distributions are stable across trips.

### Common Issues

1. **Coordinate Order:** Always lon/lat (x/y), not lat/lon.
2. **Units:** Distances in meters, angles in degrees.
3. **Missing Values:** Should not occur, but handle gracefully.
4. **Infinity/NaN:** Check for division by zero or invalid projections.

---

## References

1. **Newson & Krumm (2009):** "Hidden Markov Map Matching Through Noise and Sparseness" - foundational HMM approach for map-matching.
2. **Quddus et al. (2007):** "Current map-matching algorithms for transport applications" - comprehensive review.
3. **LightGBM Documentation:** https://lightgbm.readthedocs.io/ - feature importance and hyperparameters.

---

## Appendix: Feature Engineering Code

Feature engineering code is located in:
- **Implementation:** `src/pathsnap/features/build.py`
- **Configuration:** `configs/features.yaml`
- **Pipeline:** `pipelines/03_build_features.py`

### Example: Perpendicular Distance

```python
def perpendicular_distance(point, arc_start, arc_end):
    """
    Compute perpendicular distance from point to line segment.
    
    Args:
        point: (x, y) tuple, projected coordinates
        arc_start: (x, y) tuple
        arc_end: (x, y) tuple
    
    Returns:
        distance: perpendicular distance in meters
    """
    # Vector from arc_start to arc_end
    arc_vec = np.array(arc_end) - np.array(arc_start)
    arc_length = np.linalg.norm(arc_vec)
    
    if arc_length < 1e-6:
        # Degenerate arc (zero length)
        return np.linalg.norm(np.array(point) - np.array(arc_start))
    
    # Vector from arc_start to point
    point_vec = np.array(point) - np.array(arc_start)
    
    # Project point onto arc
    t = np.dot(point_vec, arc_vec) / (arc_length ** 2)
    t = np.clip(t, 0, 1)  # Clamp to segment
    
    # Closest point on arc
    closest = np.array(arc_start) + t * arc_vec
    
    # Distance
    return np.linalg.norm(np.array(point) - closest)
```

---

**End of Document**
