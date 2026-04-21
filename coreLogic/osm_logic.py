import gradio as gr
import matplotlib.pyplot as plt
from pyrosm import OSM
from UI.loading import render_loading

# PAGE 1 → Upload OSM
def upload_osm(file):

    if file is None:
        return (
            None,
            gr.update(visible=False),
            gr.update(),
            gr.update()
        )

    if not (file.name.endswith(".osm") or file.name.endswith(".pbf") or file.name.endswith(".osm.pbf")):
        return (
            None,
            gr.update(visible=True),
            gr.update(),
            gr.update()
        )

    filepath = file.name

    osm = OSM(filepath)
    osm.filepath = filepath

    return (
        osm,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )

# PAGE 3 → PAGE 4
def plot_highways(osm, bbox, selected_highways):

    yield render_loading(), None, None, None, gr.update(), gr.update()
    if osm is None:
        return None, None, None, gr.update(visible=True), gr.update(visible=False)

    if not selected_highways:
        return None,None, None, gr.update(visible=True), gr.update(visible=False)

    lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))

    bbox_tuple = [lon_min, lat_min, lon_max, lat_max]

    filepath = osm.filepath

    # โหลดเฉพาะพื้นที่จากไฟล์
    osm = OSM(filepath, bounding_box=bbox_tuple)

    nodes, edges = osm.get_network(nodes=True, network_type="driving")

    if edges is None or edges.empty:
        yield render_loading(), None, None, None, gr.update(visible=True), gr.update(visible=False)
        return
    highways = edges[
        edges["highway"].isin(selected_highways)
    ].copy()

    if highways.empty:
        yield render_loading(), None, None, None, gr.update(visible=True), gr.update(visible=False)
        return

    # วาดเส้นแดง
    fig, ax = plt.subplots(figsize=(8, 8))
    highways.plot(ax=ax, color="red", linewidth=1)

    ax.set_title("Selected Highways")
    ax.set_axis_off()

    print("Totel Images:", len(highways))

    yield "", fig, nodes, highways, gr.update(visible=False), gr.update(visible=True)

def go_to_page5(edges, nodes, road_name, highway_type):

    if edges is None or edges.empty:
        return None, None, None, gr.update(visible=True), gr.update(visible=False)

    target_filtered = [highway_type]

    filtered_edges = edges[
        edges["name"].str.contains(road_name, case=False, na=False) &
        edges["highway"].isin(target_filtered)
    ].copy()

    if filtered_edges.empty:
        print("ไม่พบถนน")
        return None, None, None, gr.update(visible=True), gr.update(visible=False)

    print("Road Name :", filtered_edges["name"].unique())
    print("Totel Images :", len(filtered_edges))

    node_ids = set(filtered_edges["u"]).union(set(filtered_edges["v"]))
    highway_nodes = nodes[nodes["id"].isin(node_ids)]

    fig, ax = plt.subplots(figsize=(8, 8))

    edges.plot(ax=ax, color="red", linewidth=1)
    filtered_edges.plot(ax=ax, color="blue", linewidth=2)

    ax.set_axis_off()
    plt.tight_layout()
    road_list = filtered_edges["name"].unique()
    count = len(filtered_edges)

    roads_formatted = "\n".join(road_list)

    info_text = f"""
    Road Name :
    {roads_formatted}

    Totel Images : {count}
   """

    return (
        road_name,
        highway_type,
        fig,
        info_text,
        filtered_edges,
        gr.update(visible=False),
        gr.update(visible=True)
    )