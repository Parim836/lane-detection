import osmium
import os
import uuid
import pandas as pd

import pandas as pd

def build_lane_updates(csv_path):
    df = pd.read_csv(csv_path)

    # แปลง fix_lane เป็นตัวเลข
    df["fix_lane"] = pd.to_numeric(df["fix_lane"], errors="coerce")

    lane_updates = {}

    # 🔥 group ตาม id
    for way_id, group in df.groupby("id"):

        # เอาเฉพาะที่ user เลือก
        df_fix = group[group["fix_lane"].notna()]

        # ❗ ถ้าไม่มีการเลือกเลย → ข้าม
        if len(df_fix) == 0:
            continue

        # 🔥 หา "ค่าที่ถูกเลือกมากสุด"
        counts = df_fix["fix_lane"].value_counts()
        top_values = counts[counts == counts.max()].index.tolist()

        if len(top_values) == 1:
            lane_updates[int(way_id)] = int(top_values[0])
        else:
            # 🔥 ถ้าเสมอ → เอาค่าล่าสุด
            latest_row = df_fix.sort_values("image_id").iloc[-1]
            lane_updates[int(way_id)] = int(latest_row["fix_lane"])

    return lane_updates

class LaneUpdater(osmium.SimpleHandler):

    def __init__(self, writer, lane_updates):
        super().__init__()
        self.writer = writer

        self.lane_updates = {}
        for k, v in lane_updates.items():
            try:
                self.lane_updates[int(float(k))] = v
            except:
                pass

    def node(self, n):
        self.writer.add_node(n)

    def relation(self, r):
        self.writer.add_relation(r)

    def way(self, w):

        if w.id in self.lane_updates:

            lane = str(self.lane_updates[w.id])

            tags = dict(w.tags)
            tags["lanes"] = lane

            new_way = w.replace(tags=tags)

            print(f"UPDATED WAY {w.id} → lanes = {lane}")

            self.writer.add_way(new_way)

        else:
            self.writer.add_way(w)


def update_osm_lanes(osm_path, lane_updates):

    output_dir = os.path.join(os.getcwd(), "output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #สร้างชื่อไฟล์ใหม่ทุกครั้ง
    file_id = uuid.uuid4().hex[:8]

    output_path = os.path.join(output_dir, f"lanes_updated_{file_id}.pbf")

    print("INPUT :", osm_path)
    print("OUTPUT:", output_path)

    writer = osmium.SimpleWriter(output_path)

    handler = LaneUpdater(writer, lane_updates)

    handler.apply_file(osm_path)

    writer.close()

    return output_path