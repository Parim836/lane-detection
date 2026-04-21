import gradio as gr

def go_to_highway(lon_min, lat_min, lon_max, lat_max):

    if not lon_min or not lat_min or not lon_max or not lat_max:
        bbox = "97.343,5.612,105.636,20.463"
    else:
        bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    return (
        bbox,
        build_map(bbox),
        gr.update(visible=False),
        gr.update(visible=True)
    )
def build_map(bbox=None):

    if bbox is None:
        # Thailand
        bbox = "97.343,5.612,105.636,20.463"

    return f"""
    <iframe
        src="https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker=13.75%2C100.5"
        style="width:100%; height:700px; border:none;">
    </iframe>
    """