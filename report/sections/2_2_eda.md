# 2.2 Exploratory Data Analysis & Preprocessing

## 2.2.1 Dataset Inventory and Quality Checks
Exploratory data analysis was conducted across all 80 training trips and 10 student-test trips (evaluator trips strictly excluded, as per partition rules).

All trips were systematically audited for spatial and temporal data integrity:
- **Missing Values:** Zero NaN, null, or unparseable coordinates were found in `.track`, `.nodes`, or `.arcs` across all 90 development trips.
- **Timestamp Monotonicity:** Zero non-monotonic timestamp anomalies occurred (`non_monotone_ts = 0`); all GPS observations follow strictly non-decreasing time sequences.
- **Sampling Frequency:** Most GPS observations are recorded at 1 Hz ($\Delta t = 1$ s, accounting for over 96.5% of sample intervals). However, intermittent signal loss causes temporal gaps exceeding 10 seconds (up to 300+ s in rural or canyon sections).
- **Network Topology:** Zero disconnected/isolated nodes were present in the extracted sub-graphs (`isolated_nodes = 0`). The road network sizes vary substantially per trip, spanning from 732 nodes (1,510 arcs) up to 155,003 nodes (307,392 arcs).

![Dataset Inventory Overview](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/eda_inventory.png)
*Figure 2.1: Distribution of trajectory lengths, network node/arc counts, and route lengths across dataset partitions.*

![Data Quality and Sampling Intervals](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/eda_data_quality.png)
*Figure 2.2: GPS sampling interval distributions ($\Delta t$), sensor gap frequencies, and distance from observations to reference edges.*

## 2.2.2 Metric Projection and Preprocessing
Because raw GPS coordinates are provided in angular WGS-84 (longitude, latitude) coordinates, direct Euclidean distance calculations would suffer severe latitude distortion. 

Every trip is projected into a local metric Cartesian coordinate system using an equirectangular projection centered on the trip's geographic centroid $(\lambda_0, \phi_0)$:
$$x = R_{\text{earth}} \cdot (\lambda - \lambda_0) \cdot \cos\left(\frac{\pi}{180}\phi_0\right) \cdot \frac{\pi}{180}$$
$$y = R_{\text{earth}} \cdot (\phi - \phi_0) \cdot \frac{\pi}{180}$$
with $R_{\text{earth}} = 6,371,000$ meters.

All feature engineering, spatial nearest-neighbor lookups via `scipy.spatial.cKDTree`, and point-to-segment distance computations are executed strictly within this metric space.

## 2.2.3 Candidate Selection and Miss-Rate Analysis
For each projected GPS point, candidate road arcs are extracted by querying the spatial KD-tree for the $K = 10$ nearest segment midpoints within radius $R = 80$ m. 

The empirical candidate miss rate—defined as the fraction of GPS points where the true reference arc is absent from the candidate pool—averages under 1.2% across training trips and 9.4% on student-test trips.
