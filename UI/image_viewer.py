import os
from slugify import slugify
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_images(road_name):

    slug = slugify(road_name)

    image_folder = BASE_DIR / "streetview_output" / slug

    if not image_folder.exists():
        raise FileNotFoundError(f"{image_folder} not found")

    files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    def extract_index(name):
        try:
            return int(name.split("_")[0])
        except:
            return 999999

    files = sorted(files, key=extract_index)

    images = [str(image_folder / f) for f in files]

    csv_path = BASE_DIR / f"{slug}_predictions.csv"

    lanes = {}

    if csv_path.exists():

        df = pd.read_csv(csv_path)
        lane_column = df.columns[-1]

        for _, row in df.iterrows():
            lanes[int(row["id"])] = int(row[lane_column])

    index = 0

    if not images:
        return [], 0, None, lanes, "Model Detected : ?"

    first_image = images[index]

    first_lane = lanes.get(index + 1, "?")

    lane_text = f"Model Detected : {first_lane}"

    return images, index, first_image, lanes, lane_text