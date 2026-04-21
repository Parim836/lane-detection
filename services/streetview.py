import os
import math
import requests
from PIL import Image
from io import BytesIO
from slugify import slugify
from UI.loading import render_loading

# กำหนด path หลักสำหรับเก็บรูป
BASE_FOLDER = os.path.join(os.getcwd(), "streetview_output")
os.makedirs(BASE_FOLDER, exist_ok=True)

def download_streetview(api_key, filtered_edges, road_name):
    if not api_key:
        yield "Please enter your API Key."
        return

    road_folder = slugify(road_name)
    folder = os.path.join(BASE_FOLDER, road_folder)
    os.makedirs(folder, exist_ok=True)

    def heading_from_linestring(line):
        (lon1, lat1), (lon2, lat2) = line.coords[0], line.coords[-1]
        u = lon2 - lon1
        v = lat2 - lat1
        return (math.degrees(math.atan2(u, v)) + 360) % 360

    total = len(filtered_edges)

    # แสดง loading spinner แบบไม่ต้องนับเปอร์เซ็นต์
    yield render_loading(text="Downloading images...")

    for i, geo in enumerate(filtered_edges["geometry"], start=1):
        if geo is None:
            continue

        if geo.geom_type == "MultiLineString":
            geo = max(geo.geoms, key=lambda g: g.length)

        point = geo.interpolate(0.5, normalized=True)
        lat, lon = point.y, point.x
        heading = heading_from_linestring(geo)

        url = (
            f"https://maps.googleapis.com/maps/api/streetview"
            f"?size=512x256"
            f"&location={lat},{lon}"
            f"&heading={heading:.1f}"
            f"&pitch=-10"
            f"&fov=70"
            f"&key={api_key}"
        )

        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            filename = os.path.join(folder, f"{i}_{lat:.6f}_{lon:.6f}.jpg")
            img.save(filename)

    yield ""  # ปิด loading