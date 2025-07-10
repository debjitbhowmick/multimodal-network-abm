import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import pandas as pd

# --------------------------
# Control which modal graphs to (re)build
# --------------------------
build_flags = {
    "walk": False, #False
    "bike": False,
    "pt":   False,
    "drive": False
}

# --------------------------
# Configuration
# --------------------------
place_name = "City of Melbourne, Victoria, Australia"
proj_crs = "EPSG:32755"
city_centre = (320163, 5814265)  # UTM coordinates
buffer_m = 500

# Distance thresholds for connecting transfers
walk_pt_dist_m = 50           # Walk ↔ PT
bike_pt_dist_m = 100     # Bike ↔ Train
drive_pt_dist_m = 200    # Drive ↔ Train

# --------------------------
# Set up output directories
# --------------------------
base_dir = Path(__file__).resolve().parent.parent
output_dir = base_dir / "output"
network_dir = output_dir / "networks"
figures_dir = output_dir / "figures"
multimodal_dir = output_dir / "multimodal"
for d in [network_dir, figures_dir, multimodal_dir]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------
# Custom filters for PT and parking
# --------------------------
pt_filter = '[railway~"rail|light_rail|subway|tram"]'

# --------------------------
# Simplify and save modal graphs
# --------------------------
def build_simplified_graph(mode, custom_filter=None):
    print(f"\n⏬ Downloading {mode.upper()} network...")

    if mode == "bike":
        polygon = ox.geocode_to_gdf(place_name).geometry.iloc[0]
        bike_filter_1 = '[~"highway"~"cycleway|trunk|primary|secondary|tertiary|unclassified|residential|primary_link|secondary_link|tertiary_link|living_street|trailhead|service"]["access"!~"no|private"]["bicycle"!~"no"]["area"!~"yes"]'
        bike_filter_2 = '[~"highway"~"footway|pedestrian|path"]["bicycle"~"yes|designated|dismount"]["area"!~"yes"]'
        G1 = ox.graph.graph_from_polygon(polygon, custom_filter=bike_filter_1, simplify=False, retain_all=True)
        G2 = ox.graph.graph_from_polygon(polygon, custom_filter=bike_filter_2, simplify=False, retain_all=True)
        G = nx.compose(G1, G2)
    elif custom_filter:
        G = ox.graph_from_place(place_name, custom_filter=custom_filter, simplify=False, retain_all=True)
    else:
        G = ox.graph_from_place(place_name, network_type=mode, simplify=False, retain_all=True)

    G_uns = ox.project_graph(G, to_crs=proj_crs)
    G_sim = ox.simplify_graph(G)

    if mode != "pt":
        G_sim = ox.truncate.largest_component(G_sim, strongly=False)
        G_sim = ox.project_graph(G_sim, to_crs=proj_crs)
        G_sim = ox.consolidate_intersections(G_sim, tolerance=15, rebuild_graph=True)
    else:
        G_sim = ox.project_graph(G_sim, to_crs=proj_crs)

    n_before = len(G_uns.nodes)
    e_before = len(G_uns.edges)
    n_after = len(G_sim.nodes)
    e_after = len(G_sim.edges)

    print(f"📊 {mode.upper()} Simplified: Nodes {n_before} → {n_after}, Edges {e_before} → {e_after}")

    for u, v, k, data in G_sim.edges(keys=True, data=True):
        data["mode"] = mode
        data["osmid"] = data.get("osmid", f"{mode}_edge")

    ox.save_graph_geopackage(G_sim, network_dir / f"{mode}_simplified.gpkg")
    ox.save_graphml(G_sim, network_dir / f"{mode}_simplified.graphml")

    nodes_uns, edges_uns = ox.graph_to_gdfs(G_uns, nodes=True, edges=True)
    nodes_sim, edges_sim = ox.graph_to_gdfs(G_sim, nodes=True, edges=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    nodes_uns.plot(ax=axs[0], markersize=1)
    edges_uns.plot(ax=axs[0], linewidth=0.5, color="grey")
    nodes_sim.plot(ax=axs[1], markersize=1)
    edges_sim.plot(ax=axs[1], linewidth=0.5, color="red")

    for ax in axs:
        ax.set_xlim(city_centre[0] - buffer_m, city_centre[0] + buffer_m)
        ax.set_ylim(city_centre[1] - buffer_m, city_centre[1] + buffer_m)
        ax.set_axis_off()

    axs[0].set_title(f"{mode.capitalize()} – Original")
    axs[1].set_title(f"{mode.capitalize()} – Simplified")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{mode}_comparison.png", dpi=300)
    plt.close()

    return G_sim

# --------------------------
# Get TRAIN station nodes
# --------------------------
def get_train_station_nodes():
    print("🚉 Identifying TRAIN stations for BIKE_PT and DRIVE_PT transfers...")
    return ox.features_from_place(place_name, tags={"railway": "station"}).to_crs(proj_crs)

def get_train_station_points(place_name, proj_crs):
    print("📍 Extracting train station centroids...")
    stations = ox.features_from_place(place_name, tags={"railway": "station"}).to_crs(proj_crs)
    stations["geometry"] = stations.geometry.centroid
    return stations

def connect_train_station_centroids_to_pt(G_pt, gdf_stations, max_dist=100):
    print("🔗 Connecting TRAIN station centroids to nearest PT nodes...")
    
    # Project PT nodes (if not already)
    gdf_pt_nodes = ox.graph_to_gdfs(G_pt, nodes=True, edges=False)
    gdf_pt_nodes["node_id"] = gdf_pt_nodes.index

    # Nearest PT node for each station centroid
    matches = gpd.sjoin_nearest(gdf_stations, gdf_pt_nodes, how="inner", max_distance=max_dist)

    for _, row in matches.iterrows():
        station_geom = row.geometry
        pt_node_id = row["node_id"]
        pt_geom = gdf_pt_nodes.loc[pt_node_id].geometry
        dist = station_geom.distance(pt_geom)
        if dist <= max_dist:
            # Add a node for the station if not already in G_combined
            station_id = f"station_{pt_node_id}"  # unique station node ID
            if station_id not in G_combined:
                G_combined.add_node(station_id, geometry=station_geom, x=station_geom.x, y=station_geom.y)

            # Add edge from station to PT node
            G_combined.add_edge(
                station_id, pt_node_id,
                mode="station_to_pt",
                length=dist,
                travel_time=int(dist / 1.4),  # assuming walking
                osmid="station_connector"
            )


# --------------------------
# Connect WALK ↔ PT transfers
# --------------------------
def connect_walk_to_pt_stops(G_walk, G_pt, pt_dist_m):
    print("🚶 Connecting WALK ↔ PT using PT stop centroids...")

    tags = {
        "public_transport": ["platform", "stop_position", "station"],
        "railway": ["station", "tram_stop"],
        "highway": "bus_stop"
    }
    gdf_stops = ox.features_from_place(place_name, tags=tags).to_crs(proj_crs)
    gdf_stops["geometry"] = gdf_stops.geometry.centroid
    gdf_stops = gdf_stops.reset_index(drop=True)
    gdf_stops["stop_id"] = ["ptstop_" + str(i) for i in gdf_stops.index]

    walk_nodes = ox.graph_to_gdfs(G_walk, nodes=True, edges=False).to_crs(proj_crs)
    walk_nodes["node_id"] = walk_nodes.index

    nearest_walk = gpd.sjoin_nearest(gdf_stops, walk_nodes, how="inner", max_distance=pt_dist_m)

    for _, row in nearest_walk.iterrows():
        stop_geom = row.geometry
        stop_id = row["stop_id"]
        walk_id = row["node_id"]
        walk_geom = walk_nodes.loc[walk_id].geometry
        dist = stop_geom.distance(walk_geom)

        if dist <= pt_dist_m:
            x, y = stop_geom.x, stop_geom.y
            G_combined.add_node(stop_id, geometry=stop_geom, x=x, y=y, mode_set={"walk_pt_transfer"})
            G_combined.add_edge(
                walk_id, stop_id,
                mode="walk_pt_transfer",
                length=dist,
                travel_time=int(dist / 1.4),
                osmid="walk_pt_connector"
            )

# --------------------------
# Connect BIKE ↔ PT using TRAIN stations only
# --------------------------
def connect_bike_to_train_stations(G_bike, G_pt, bike_pt_dist_m):
    print("🚴 Connecting BIKE ↔ PT using TRAIN station centroids...")

    gdf_stations = get_train_station_points(place_name, proj_crs)
    connect_train_station_centroids_to_pt(G_pt, gdf_stations)
    gdf_stations = gdf_stations.reset_index(drop=True)
    gdf_stations["station_id"] = ["station_" + str(i) for i in gdf_stations.index]
    # SAVING THE TRAIN STATIONS AS A GPKG FILE
    # Clean field names (replace ':' with '_')
    gdf_stations_clean = gdf_stations.copy()
    gdf_stations_clean.columns = [col.replace(":", "_") for col in gdf_stations_clean.columns]
    # Keep only safe fields (add more as needed)
    safe_fields = ["geometry"]
    if "name" in gdf_stations_clean.columns:
        safe_fields.append("name")
    if "station_id" in gdf_stations_clean.columns:
        safe_fields.append("station_id")
    gdf_stations_safe = gdf_stations_clean[safe_fields]
    # Save to GPKG
    gdf_stations_safe.to_file(network_dir / "train_station_centroids.gpkg", driver="GPKG")


    bike_nodes = ox.graph_to_gdfs(G_bike, nodes=True, edges=False).to_crs(proj_crs)
    bike_nodes["node_id"] = bike_nodes.index

    nearest_bike = gpd.sjoin_nearest(gdf_stations, bike_nodes, how="inner", max_distance=bike_pt_dist_m)

    for _, row in nearest_bike.iterrows():
        station_geom = row.geometry
        station_id = row["station_id"]
        bike_id = row["node_id"]
        bike_geom = bike_nodes.loc[bike_id].geometry
        dist = station_geom.distance(bike_geom)

        if dist <= bike_pt_dist_m:
            x, y = station_geom.x, station_geom.y
            G_combined.add_node(station_id, geometry=station_geom, x=x, y=y, mode_set={"bike_pt_transfer"})
            G_combined.add_edge(
                bike_id, station_id,
                mode="bike_pt_transfer",
                length=dist,
                travel_time=int(dist / 3.0),
                osmid="bike_pt_connector"
            )


# --------------------------
# Connect DRIVE ↔ PT using TRAIN stations only
# --------------------------
def connect_drive_to_train_stations(G_drive, G_pt, drive_pt_dist_m):
    print("🅿️ Connecting DRIVE ↔ PT using TRAIN station centroids...")

    gdf_stations = get_train_station_points(place_name, proj_crs)
    connect_train_station_centroids_to_pt(G_pt, gdf_stations)
    gdf_stations = gdf_stations.reset_index(drop=True)
    gdf_stations["station_id"] = ["station_" + str(i) for i in gdf_stations.index]
    # gdf_stations.to_file(network_dir / "train_station_centroids.gpkg", driver="GPKG")

    drive_nodes = ox.graph_to_gdfs(G_drive, nodes=True, edges=False).to_crs(proj_crs)
    drive_nodes["node_id"] = drive_nodes.index

    nearest_drive = gpd.sjoin_nearest(gdf_stations, drive_nodes, how="inner", max_distance=drive_pt_dist_m)

    for _, row in nearest_drive.iterrows():
        station_geom = row.geometry
        station_id = row["station_id"]
        drive_id = row["node_id"]
        drive_geom = drive_nodes.loc[drive_id].geometry
        dist = station_geom.distance(drive_geom)

        if dist <= drive_pt_dist_m:
            x, y = station_geom.x, station_geom.y
            G_combined.add_node(station_id, geometry=station_geom, x=x, y=y, mode_set={"drive_pt_transfer"})
            G_combined.add_edge(
                drive_id, station_id,
                mode="drive_pt_transfer",
                length=dist,
                travel_time=int(dist / 2.5),
                osmid="drive_pt_connector"
            )


# --------------------------
# Annotate each node with available modes
# --------------------------
def tag_node_modes(G):
    for node in G.nodes():
        G.nodes[node]["mode_set"] = set()
    for u, v, data in G.edges(data=True):
        mode = data.get("mode")
        if mode:
            G.nodes[u]["mode_set"].add(mode)
            G.nodes[v]["mode_set"].add(mode)

def ensure_fully_connected_graph(G, max_connector_dist=100):
    """
    Ensures the graph is fully connected by linking smaller components
    to the largest one using nearest neighbour edges.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The final combined graph.
    max_connector_dist : float
        Max distance to connect isolated components (in metres).

    Returns
    -------
    networkx.MultiDiGraph
        A fully connected graph.
    """
    import networkx as nx
    import osmnx as ox
    import geopandas as gpd

    print("🔍 Checking connectivity of the final multimodal graph...")

    # Get projected node GeoDataFrame
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()

    # Check if graph is projected
    if gdf_nodes.crs is None or not gdf_nodes.crs.is_projected:
        raise ValueError("❌ Graph is not projected. Please project it before running this function.")

    # Ensure unique node_id
    gdf_nodes["node_id"] = gdf_nodes["osmid"]

    # Check connectivity
    components = list(nx.connected_components(G.to_undirected()))
    num_components = len(components)

    if num_components == 1:
        print("✅ The network is already fully connected.")
        return G

    print(f"⚠️ The network is NOT fully connected. It has {num_components} components.")
    print("🛠 Attempting to repair by connecting smaller components...")

    # Largest component
    components.sort(key=len, reverse=True)
    largest_component = components[0]
    gdf_largest = gdf_nodes[gdf_nodes["osmid"].isin(largest_component)].copy()
    gdf_largest = gdf_largest.rename(columns={"osmid": "target_osmid"})

    for i in range(1, num_components):
        comp_nodes = list(components[i])
        gdf_small = gdf_nodes[gdf_nodes["osmid"].isin(comp_nodes)].copy()
        gdf_small = gdf_small.rename(columns={"osmid": "source_osmid"})

        try:
            joined = gpd.sjoin_nearest(
                gdf_small[["source_osmid", "geometry"]],
                gdf_largest[["target_osmid", "geometry"]],
                how="left",
                max_distance=max_connector_dist,
                distance_col="dist"
            )
        except Exception as e:
            print(f"❌ Failed to connect component {i}: {e}")
            continue

        for _, row in joined.iterrows():
            u = row["source_osmid"]
            v = row["target_osmid"]
            dist = row["dist"]

            if pd.isna(dist):
                print(f"⚠️ Skipping unmatched node {u} in component {i} (no nearby node within {max_connector_dist} m)")
                continue

            G.add_edge(u, v,
                       mode="repair_connector",
                       length=dist,
                       travel_time=int(dist / 1.4),
                       osmid="repair")
            G.add_edge(v, u,
                       mode="repair_connector",
                       length=dist,
                       travel_time=int(dist / 1.4),
                       osmid="repair")


    # Final connectivity check
    final_components = nx.number_connected_components(G.to_undirected())
    if final_components == 1:
        print("✅ The network is now fully connected!")
    else:
        print(f"⚠️ Still has {final_components} components after repair.")

    return G




# --------------------------
# MAIN SCRIPT
# --------------------------
if __name__ == "__main__":
    print("🏗 Building mode-specific simplified graphs...")
    G_pt   = build_simplified_graph("pt", custom_filter=pt_filter) if build_flags["pt"] else ox.load_graphml(network_dir / "pt_simplified.graphml")
    G_walk = build_simplified_graph("walk") if build_flags["walk"] else ox.load_graphml(network_dir / "walk_simplified.graphml")
    G_bike = build_simplified_graph("bike") if build_flags["bike"] else ox.load_graphml(network_dir / "bike_simplified.graphml")
    G_drive = build_simplified_graph("drive") if build_flags["drive"] else ox.load_graphml(network_dir / "drive_simplified.graphml")

    G_walk = nx.relabel_nodes(G_walk, lambda x: f"walk_{x}")
    G_bike = nx.relabel_nodes(G_bike, lambda x: f"bike_{x}")
    G_pt = nx.relabel_nodes(G_pt, lambda x: f"pt_{x}")
    G_drive = nx.relabel_nodes(G_drive, lambda x: f"drive_{x}")

    print("\n📦 Combining graphs...")
    # G_combined = nx.compose(G_bike, G_walk)
    G_combined = nx.compose_all([G_pt, G_drive, G_bike, G_walk])

    connect_walk_to_pt_stops(G_walk, G_pt, walk_pt_dist_m)
    connect_bike_to_train_stations(G_bike, G_pt, bike_pt_dist_m)
    connect_drive_to_train_stations(G_drive, G_pt, drive_pt_dist_m)

    tag_node_modes(G_combined)

    # ox.save_graphml(G_combined, multimodal_dir / "melbourne_multimodal.graphml")
    ox.save_graph_geopackage(G_combined, multimodal_dir / "melbourne_multimodal.gpkg")
    print("✅ Final multimodal graph saved.")

    print("🔍 Checking connectivity of the final multimodal graph...")

    if nx.is_connected(G_combined.to_undirected()):
        print("✅ The network is fully connected.")
    else:
        num_components = nx.number_connected_components(G_combined.to_undirected())
        print(f"⚠️ The network is NOT fully connected. It has {num_components} components.")
        
        # Optional: find the largest connected component
        largest_cc = max(nx.connected_components(G_combined.to_undirected()), key=len)
        print(f"🧩 Largest component has {len(largest_cc)} nodes out of {G_combined.number_of_nodes()} total.")

    G_combined_connected = ensure_fully_connected_graph(G_combined, max_connector_dist=200)
    ox.save_graph_geopackage(G_combined_connected, multimodal_dir / "melbourne_multimodal_connected.gpkg")
    print("✅ Final connected multimodal graph saved.")


