# 6. Interactive Route Visualisations

## 6.1 Folium Map Architecture
To inspect and explain map-matching predictions during the viva evaluation, we implemented an interactive mapping module in `viz/maps/folium_maps.py`. The visualizer renders multi-layer interactive maps using Folium / Leaflet.js with LayerControl toggles.

### Map Layers and Legend
1. **Background Base Map:** CartoDB Positron / OpenStreetMap standard tiles providing cartographic context.
2. **Road Network Arcs (Gray `#aaaaaa`):** All directed arcs in the trip's local road network graph, displaying edge connectivity and available candidate alternatives.
3. **GPS Observations (Blue Circles `#1f77ff`):** Raw, noisy GPS points plotted with circular markers showing sequence index, sensor timestamps, and speed.
4. **Reference Ground Truth (Green `#2ca02c`, weight 5):** The official reference sequence of traversed road arcs.
5. **Predicted Path (Red `#e41a1c`, weight 3.5):** The final route reconstructed by the XGBoost candidate scorer and HMM/Viterbi sequence decoder.

![Road Network and Trajectories](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/report/figures/eda_trajectories_network.png)
*Figure 6.1: Overview of spatial road network topology, candidate corridors, and reconstructed trajectory sequences.*

## 6.2 Generated Interactive Maps
Self-contained HTML maps are generated in `maps/` for offline inspection:
- [trip_student_test_00000007.html](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/maps/trip_student_test_00000007.html): Demonstrates seamless sequence reconstruction along multi-lane arterial roads.
- [trip_student_test_00000009.html](file:///c:/Users/punit/OneDrive%20-%20PESUNIVERSITY/Desktop/ml_hackathon/maps/trip_student_test_00000009.html): Demonstrates correct junction disambiguation where raw GPS noise deviated into intersecting side streets.

To generate interactive maps for any trip:
```bash
python viz/maps/folium_maps.py data/raw/student_test 00000007
```
