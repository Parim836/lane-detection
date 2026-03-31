import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2
import csv

BASE_DIR = Path(__file__).resolve().parent



def cluster_lanes(binary, min_area=200):

    if binary.max() <= 1:
        binary = (binary * 255).astype(np.uint8)
    else:
        binary = binary.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    kernel = np.ones((4,4), np.uint8)
    dilated = cv2.dilate(filtered, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    num_labels, labels_im = cv2.connectedComponents(eroded, connectivity=8)

    lanes = num_labels - 2

    return max(lanes, 0)


def csv_file(binary_folder, output_csv):

    fields = ['id', 'lon', 'lat', 'lanes']

    image_files = [
        f for f in os.listdir(binary_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    with open(output_csv, 'w', newline='', encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for file in image_files:

            image_path = os.path.join(binary_folder, file)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("Failed to read:", image_path)
                continue

            binary = (img > 127).astype(np.uint8)

            num_lanes = cluster_lanes(binary)

            filename_no_ext = os.path.splitext(file)[0]

            try:

                parts = filename_no_ext.split("_")

                # filename format: id_lat_lon.jpg
                image_id = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])

            except:
                print("Filename format error:", file)
                continue

            writer.writerow([
                image_id,
                lon,
                lat,
                num_lanes
            ])

    sorted_id_csv(output_csv)

    print("Complete", output_csv)
def sorted_id_csv(csv_name):

    df = pd.read_csv(fr"{BASE_DIR}\{csv_name}")

    df['id'] = df['id'].astype(int)

    df = df.sort_values(by='id').reset_index(drop=True)

    df.to_csv(csv_name, index=False)


def combine_binary(folder_name):

    files = os.listdir(fr"{BASE_DIR}\Model\unet\output\{folder_name}")

    output_folder = fr"{BASE_DIR}\Output_Binary\{folder_name}"

    os.makedirs(output_folder, exist_ok=True)

    for f in files:

        img1 = cv2.imread(fr"{BASE_DIR}\Model\lanenet-lane-detection\output\{folder_name}\{f}", 0)
        _, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

        img2 = cv2.imread(fr"{BASE_DIR}\Model\lanenet-lane-detection-pytorch\output\{folder_name}\{f}", 0)
        _, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        img3 = cv2.imread(fr"{BASE_DIR}\Model\deeplabv3\output\{folder_name}\{f}", 0)
        _, img3 = cv2.threshold(img3, 127, 255, cv2.THRESH_BINARY)

        img4 = cv2.imread(fr"{BASE_DIR}\Model\unet\output\{folder_name}\{f}", 0)
        _, img4 = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)

        img1 = (img1 == 255).astype(np.uint8)
        img2 = (img2 == 255).astype(np.uint8)
        img3 = (img3 == 255).astype(np.uint8)
        img4 = (img4 == 255).astype(np.uint8)

        stack = np.stack([img1, img2, img3, img4])

        vote = np.sum(stack, axis=0)

        result = (vote >= 2).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(output_folder, f), result)

    # สร้าง csv
    csv_file(
        binary_folder=BASE_DIR / "Output_Binary" / folder_name,
        output_csv=f"{folder_name}_predictions.csv"
    )

    print("Complete combine")


def merge_osm_data(pred_csv, filtered_edges, output_csv):

    import pandas as pd
    import os

    pred = pd.read_csv(pred_csv)

    # เปลี่ยนชื่อ column prediction
    pred = pred.rename(columns={"lanes": "pred_lanes"})

    # ใช้เฉพาะจำนวน prediction เท่านั้น
    n = len(pred)

    df = filtered_edges.iloc[:n].copy()

    print("Rows used:", len(df))

    # คำนวณ centroid
    df_proj = df.to_crs(3857)
    centroids = df_proj.geometry.centroid.to_crs(4326)

    df["lon"] = centroids.x
    df["lat"] = centroids.y

    # lane จาก OSM
    df["osm_lanes"] = pd.to_numeric(df["lanes"], errors="coerce")

    # รวม prediction
    df["pred_lanes"] = pred["pred_lanes"].values

    # dataframe output
    df_out = pd.DataFrame({
        "id": df["id"].astype(str),
        "lon": df["lon"],
        "lat": df["lat"],
        "osm_lanes": df["osm_lanes"],
        "pred_lanes": df["pred_lanes"]
    })

    df_out["fix_lane"] = ""

    # error ถ้าไฟล์เปิดอยู่
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except:
            print("Warning: cannot overwrite file (maybe opened in Excel)")

    df_out.to_csv(output_csv, index=False)

    print("Saved:", output_csv)