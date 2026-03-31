from model import LaneNet_TensorFlow, LaneNet_PyTorch, DeepLabV3, UNet
from count import combine_binary, merge_osm_data
from slugify import slugify
import os
import pandas as pd


def run_lane_pipeline(road_name, filtered_edges):

    road_slug = slugify(road_name)

    print("Running LaneNet TensorFlow...")
    LaneNet_TensorFlow(road_slug)

    print("Running LaneNet PyTorch...")
    LaneNet_PyTorch(road_slug)

    print("Running DeepLabV3...")
    DeepLabV3(road_slug)

    print("Running UNet...")
    UNet(road_slug)

    print("Combining results...")
    combine_binary(road_slug)

    print("Lane detection complete")

    pred_csv = f"{road_slug}_predictions.csv"
    output_csv = f"{road_slug}_lanes.csv"

    if not os.path.exists(pred_csv):
        print("CSV not found:", pred_csv)
        return None

    merge_osm_data(
        pred_csv=pred_csv,
        filtered_edges=filtered_edges,
        output_csv=output_csv
    )

    # ⭐ เพิ่ม column สำหรับ UI
    df = pd.read_csv(output_csv)

    df["image_id"] = range(1, len(df) + 1)
    df["fix_lane"] = None

    df.to_csv(output_csv, index=False)

    return output_csv