
from pathlib import Path
import pandas as pd
import folium

TRIP_ID = "00000001"
TRIP_DIR = Path("data/raw/Student folder/train") / TRIP_ID

# Load dataset
track = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.track",
    sep=r"\s+", header=None,
    names=["longitude", "latitude", "timestamp"]
)

nodes = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.nodes",
    sep=r"\s+", header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.arcs",
    sep=r"\s+", header=None,
    names=["from_node", "to_node"]
)

route = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.route",
    sep=r"\s+", header=None,
    names=["arc_id"]
)

# Map centred on GPS trajectory
center = [
    track["latitude"].mean(),
    track["longitude"].mean()
]

m = folium.Map(location=center, zoom_start=11)

# 1. GPS trajectory
gps_layer = folium.FeatureGroup(
    name="GPS trajectory (Blue)",
    show=True
)

gps_points = list(zip(track["latitude"], track["longitude"]))

folium.PolyLine(
    gps_points,
    color="blue",
    weight=3,
    opacity=0.9,
    tooltip="GPS trajectory"
).add_to(gps_layer)

gps_layer.add_to(m)

# 2. Ground-truth route
route_layer = folium.FeatureGroup(
    name="Ground-truth route (Red)",
    show=True
)

for arc_id in route["arc_id"]:
    arc = arcs.iloc[int(arc_id)]

    start = nodes.iloc[int(arc["from_node"])]
    end = nodes.iloc[int(arc["to_node"])]

    folium.PolyLine(
        [
            [start["latitude"], start["longitude"]],
            [end["latitude"], end["longitude"]]
        ],
        color="red",
        weight=4,
        opacity=0.85
    ).add_to(route_layer)

route_layer.add_to(m)

# 3. Nearby road network
road_layer = folium.FeatureGroup(
    name="Nearby road network (Grey)",
    show=True
)

# Small geographic margin around the GPS trajectory
margin = 0.02

min_lon = track["longitude"].min() - margin
max_lon = track["longitude"].max() + margin
min_lat = track["latitude"].min() - margin
max_lat = track["latitude"].max() + margin

# Select nodes inside the map area
inside = (
    nodes["longitude"].between(min_lon, max_lon)
    & nodes["latitude"].between(min_lat, max_lat)
)

# Keep only arcs whose two endpoints are inside the area
nearby_arcs = arcs[
    inside.iloc[arcs["from_node"].to_numpy()].to_numpy()
    & inside.iloc[arcs["to_node"].to_numpy()].to_numpy()
]

for arc in nearby_arcs.itertuples(index=False):
    start = nodes.iloc[int(arc.from_node)]
    end = nodes.iloc[int(arc.to_node)]

    folium.PolyLine(
        [
            [start["latitude"], start["longitude"]],
            [end["latitude"], end["longitude"]]
        ],
        color="grey",
        weight=1,
        opacity=0.45
    ).add_to(road_layer)

road_layer.add_to(m)

# Layer control
folium.LayerControl(collapsed=False).add_to(m)

m.fit_bounds([
    [track["latitude"].min(), track["longitude"].min()],
    [track["latitude"].max(), track["longitude"].max()]
])

# Save map
output_dir = Path("visualisations")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f"{TRIP_ID}_map.html"
m.save(output_file)

print(f"Map saved: {output_file}")
print(f"Total road arcs: {len(arcs)}")
print(f"Nearby road arcs displayed: {len(nearby_arcs)}")
print(f"GPS observations: {len(track)}")
print(f"Ground-truth route arcs: {len(route)}")
