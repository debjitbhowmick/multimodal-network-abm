# 🗺️ Multimodal Transport Network Builder for Greater Melbourne

This project constructs a **fully connected multimodal transport network** for Greater Melbourne by combining simplified mode-specific graphs (walking, biking, driving, and public transport) using [OSMnx](https://github.com/gboeing/osmnx) and OpenStreetMap data.

---

## 🚀 Features

- 🔧 Build simplified graphs for each mode: **walk**, **bike**, **drive**, **PT**
- 🔄 Project networks into a common **metric CRS**
- 🔗 Connect modes at logical intermodal points:
  - WALK ↔ PT: PT stop centroids
  - BIKE/DRIVE ↔ PT: Train station centroids
- 🧠 Automatically **connect disconnected components** in the final graph
- 💾 Save final network as `.graphml` and `.gpkg`
- 📍 Export train station centroids as GeoPackage

---

## 🧰 Dependencies

Install via `conda` or `pip`:

```bash
conda install -c conda-forge osmnx networkx geopandas pyproj shapely pandas

pip install osmnx networkx geopandas pyproj shapely pandas
```

## 🛠 How It Works
Simplify and project individual networks for each mode.

Extract PT stop and train station centroids.

Connect:

WALK ↔ PT using PT stops

BIKE ↔ PT and DRIVE ↔ PT using train station centroids

Merge all mode-specific graphs into one unified network.

Ensure full connectivity by linking smaller components to the largest one using nearest-neighbour logic.

Save output in multiple formats for downstream use in routing or simulation models (e.g., MATSim).

## ✅ Outputs
Output File	Description
melbourne_multimodal.graphml	Final connected multimodal graph
melbourne_multimodal.gpkg	Same graph as GeoPackage
train_station_centroids.gpkg	Cleaned train station centroid points used for intermodal connection
