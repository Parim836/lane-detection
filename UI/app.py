from coreLogic.osm_logic import upload_osm, plot_highways, go_to_page5
from services.streetview import download_streetview
from map import go_to_highway, build_map
from image_viewer import load_images
from coreLogic.pipeline import run_lane_pipeline
from loading import render_loading
from coreLogic.osm_update import update_osm_lanes
import gradio as gr
import pandas as pd
import os

with open("style.css", encoding="utf-8") as f:
    css = f.read()
fix_lanes = {}
def handle_all(selected):

        highway_list = [
            "secondary",
            "primary",
            "primary_link",
            "secondary_link",
            "motorway_link",
            "trunk",
            "motorway",
            "trunk_link"
        ]

        # ถ้าเลือก all
        if "all" in selected:
            return highway_list + ["all"]

        # ถ้าเอา all ออก
        if "all" not in selected and len(selected) == len(highway_list):
            return []

        return selected
def update_highway_options(edges, road):

        if edges is None or road is None:
            return gr.update(choices=[])

        df = edges[edges["name"] == road]

        if df.empty:
            return gr.update(choices=[])

        types = (
            df["highway"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        types.sort()

        return gr.update(
            choices=types,
            value=types[0] if len(types) > 0 else None
        )
with gr.Blocks(css=css) as demo: 
      
    osm_state = gr.State(None)
    bbox_state = gr.State("")
    nodes_state = gr.State()
    edges_state = gr.State()
    road_name_state = gr.State()
    filtered_edges_state = gr.State()
    highway_type_state = gr.State()
    images_state = gr.State([])
    index_state = gr.State(0)
    lane_state = gr.State({})
    sidebar_state = gr.State(False)
    selected_data = gr.State([])
    csv_path_state = gr.State()
    updated_osm_state = gr.State()
    
    # PAGE 1 : Upload
    with gr.Column(visible=True) as page_upload:

        gr.Markdown("# Read File OpenStreetMap (OSM)", elem_id="text_page1")

        file_input = gr.File(label="Only .osm and .pbf files are supported",elem_id="textupload_box")
        upload_btn = gr.Button("Upload File", elem_id="upload_btn")
        
        error_modal = gr.HTML(
        """
        <div class="modal_overlay" onclick="this.style.display='none'">
            <div class="modal_card" onclick="event.stopPropagation()">
                <div class="modal_icon">❌</div>
                <div class="modal_text">
                    File upload failed.<br>
                    Please upload the correct OSM file.<br>
                    Only .osm and .pbf files are supported
                </div>
            </div>
        </div>
        """,
        visible=False
        )
        
        upload_status = gr.Markdown()
    def check_upload(file):

        if file is None:
            return (
                None,
                "<script>document.getElementById('error_overlay').style.display='flex'</script>",
                gr.update(),
                gr.update()
            )

        if not file.name.endswith(".osm"):
            return (
                None,
                "<script>document.getElementById('error_overlay').style.display='flex'</script>",
                gr.update(),
                gr.update()
            )

        osm_obj = upload_osm(file)

        return (
            osm_obj,
            "",
            gr.update(visible=False),
            gr.update(visible=True)
        )

    # PAGE 2 : Bounding Box
    with gr.Column(visible=False) as page_bbox:

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("## Bounding Box")
                lon_min = gr.Textbox(label="Longitude Min",value="100.48164")
                lat_min = gr.Textbox(label="Latitude Min",value="13.78332")
                lon_max = gr.Textbox(label="Longitude Max",value="100.49512")
                lat_max = gr.Textbox(label="Latitude Max",value="13.79070")

                btn1 = gr.Button("Search", elem_id="download_btn")

            with gr.Column(scale=4):
                
                map_view_bbox = gr.HTML(build_map())
                loading_overlay2 = gr.HTML("")

    # PAGE 3 : Network + Highway
    with gr.Column(visible=False) as page_highway:

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("## Target Highway")

                highway = gr.CheckboxGroup([
                    "secondary",
                    "primary",
                    "primary_link",
                    "secondary_link",
                    "motorway_link",
                    "trunk",
                    "motorway",
                    "trunk_link",
                    "all"
                ])
                highway.change(
                    handle_all,
                    inputs=highway,
                    outputs=highway
                )
                

                btn2 = gr.Button("Search Highway", elem_id="download_btn")

            with gr.Column(scale=4):
                map_view_scope = gr.HTML()   #แผนที่ออนไลน์ตาม bbox
                loading_overlay2 = gr.HTML("")

    # PAGE 4 : Filter Input
    with gr.Column(visible=False) as page_result:

        with gr.Row():

            with gr.Column(scale=1, elem_classes="final-card"):

                gr.Markdown("## Filter By Road Name")

                road_name = gr.Dropdown(
                    label="Road Name",
                    choices=[],
                    interactive=True
                )

                highway_type_input = gr.Dropdown(
                    label="Highway Type",
                    choices=[],
                    interactive=True
                )
                road_name.change(
                    update_highway_options,
                    inputs=[edges_state, road_name],
                    outputs=highway_type_input
                )

                btn3 = gr.Button("Next", elem_id="download_btn")

            with gr.Column(scale=4, elem_classes="map-card"):
                result_map = gr.Plot()

    # PAGE 5 : Final Result
    with gr.Column(visible=False, elem_id="page_final") as page_final:

        with gr.Row():

            with gr.Column(scale=1, elem_classes="final-card"):
                gr.Markdown("## Final Result")
                result_info = gr.Markdown()

                gr.Markdown("## Download images from Google Street View Static API")
                apikey_input = gr.Textbox(
                    label="API Key",
                    placeholder="เช่น 1234567890"
                )

                btn4 = gr.Button("Download Picture", elem_id="download_btn")
                back_to_filter_btn = gr.Button("Back", elem_id="downloadback")

            with gr.Column(scale=4, elem_classes="map-card"):
                final_map = gr.Plot()
                loading_overlay = gr.HTML("")

    # PAGE 6 : page_show
    with gr.Column(visible=False, elem_id="page_show") as page_show:

        with gr.Column(elem_id="content_wrapper") as content_wrapper:

            sidebar_btn = gr.Button("Selected List", elem_id="floating_btn")

            gr.Markdown("# Street View Lane Viewer")

            counter = gr.Markdown("Image : 0 / 0", elem_classes="counter-text")

            image_view = gr.Image(height=400, show_label=False)

            lane_text = gr.Markdown("Model Detected : ?", elem_classes="lane-text")

            with gr.Row():
                viewer_back_btn = gr.Button("Prev", elem_id="back_btn")
                select_btn = gr.Button("Select", elem_id="select_btn")
                skip_btn = gr.Button("Next", elem_id="skip_btn")

        with gr.Column(visible=False, elem_id="sidebar_panel") as sidebar_panel:

            gr.Markdown("## Selected Data")

            selected_table = gr.Dataframe(
                headers=["image","lane",""],
                value=[],
                interactive=False,
                elem_id="selected_table"
            )
            with gr.Column(elem_id="sidebar_bottom"):
                submit_btn = gr.Button("Submit", elem_id="download_btn")
                status_text = gr.HTML("")
    with gr.Column(visible=False, elem_id="page_download") as page_download:

        with gr.Column(elem_id="download_card"):

            gr.Markdown(
                """
                <div class="download_title">
                Download Updated OpenStreetMap
                </div>
                """,
                elem_id="title_box"
            )

            gr.Markdown(
                "Your OpenStreetMap file has been updated with lane information."
            )

            download_button = gr.HTML()
    images = []
    current_index = 0
    def go_back(images, index, lanes, selected):

        global fix_lanes

        if index > 0:
            index -= 1

            image_id = index + 1

            # ลบจาก fix_lanes
            if image_id in fix_lanes:
                del fix_lanes[image_id]

            # ลบสิ่งที่เลือกจากในlist
            selected = [
                row for row in selected
                if int(row[0].split("_")[0]) != image_id
            ]

        img = images[index]
        lane = lanes.get(index + 1, "?")

        lane_text = f"Model Detected : {lane}"
        counter = f"Image : {index+1} / {len(images)}"

        return (
            index,
            img,
            lane_text,
            counter,
            selected,
            selected,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    def extract_road_names(highways):

        if highways is None or highways.empty:
            return gr.update(choices=[])

        roads = (
            highways["name"]
            .dropna()
            .unique()
            .tolist()
        )

        roads.sort()

        return gr.update(choices=roads)
    def make_download_button(path):

        filename = os.path.basename(path)

        return f"""
        <a href="file={path}" download style="text-decoration:none;">
            <div class="big_download_btn">
                ⬇ Download 
            </div>
        </a>
        """
    def process_highway(selected):

        if "all" in selected:
            selected = [
                "secondary",
                "primary",
                "primary_link",
                "secondary_link",
                "motorway_link",
                "trunk",
                "motorway",
                "trunk_link",
            ]
        return selected
    
    def get_download(path):
        return path
    def back_to_filter():
        return gr.update(visible=False), gr.update(visible=True)
                                
    def show_page():
        return gr.update(visible=False), gr.update(visible=True)

    def toggle_sidebar(state):

        state = not state

        if state:
            return (
                state,
                gr.update(visible=True),
                gr.update(elem_classes=["shift-left"])
            )
        else:
            return (
                state,
                gr.update(visible=False),
                gr.update(elem_classes=[])
            )

    def select_image(selected, images, index, lanes):

        global fix_lanes

        img = images[index]
        filename = os.path.basename(img)

        lane = lanes.get(index + 1, "?")

        image_id = index + 1
        fix_lanes[image_id] = lane

        selected.append([filename, lane, "❌"])
        selected.sort(key=lambda x: int(x[0].split("_")[0]))

        index += 1

        # ถ้ารูปหมด
        if index >= len(images):
            return (
                selected,
                selected,
                index-1,
                None,
                "Finished",
                f"Image : {len(images)} / {len(images)}",
                gr.update(interactive=False), 
                gr.update(interactive=False) 
            )

        next_img = images[index]
        lane_next = lanes.get(index + 1, "?")

        lane_text = f"Model Detected : {lane_next}"
        counter = f"Image : {index+1} / {len(images)}"

        return (
            selected,
            selected,
            index,
            next_img,
            lane_text,
            counter,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

    def skip_image(images, index, lanes):

        index += 1

        if index >= len(images):
            return (
                index-1,
                None,
                "Finished",
                f"Image : {len(images)} / {len(images)}",
                gr.update(interactive=False),
                gr.update(interactive=False)
            )

        img = images[index]
        lane = lanes.get(index + 1, "?")

        lane_text = f"Model Detected : {lane}"
        counter = f"Image : {index+1} / {len(images)}"

        return (
            index,
            img,
            lane_text,
            counter,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    def submit_fix(osm_obj, csv_path):
        global fix_lanes

        df = pd.read_csv(csv_path, dtype={"id": str})

        # ใส่ค่าที่ user เลือก
        for img_id, lane in fix_lanes.items():
            df.loc[df["image_id"] == img_id, "fix_lane"] = lane

        df.to_csv(csv_path, index=False)

        lane_updates = {}

        for way_id, g in df.groupby("id"):

            lane = None

            # ค่าใหม่จาก user (mode)
            new_lane = None
            if g["fix_lane"].notna().any():
                selected = g["fix_lane"].dropna().astype(int)

                mode_val = selected.mode()
                if not mode_val.empty:
                    new_lane = int(mode_val.iloc[0])
                    print(f"[WAY {way_id}] new_lane (mode) = {new_lane}")

            # ค่าเดิมจาก osm (mode)
            osm_lane = None
            if g["osm_lanes"].notna().any():
                osm_vals = g["osm_lanes"].dropna().astype(float).astype(int)

                mode_osm = osm_vals.mode()
                if not mode_osm.empty:
                    osm_lane = int(mode_osm.iloc[0])
                    print(f"[WAY {way_id}] osm_lane = {osm_lane}")

            if new_lane is not None and osm_lane is not None:
                lane = max(new_lane, osm_lane)
                print(f"[WAY {way_id}] use max(new, osm) = {lane}")

            elif new_lane is not None:
                lane = new_lane
                print(f"[WAY {way_id}] use new_lane only = {lane}")

            elif osm_lane is not None:
                lane = osm_lane
                print(f"[WAY {way_id}] use osm_lane only = {lane}")

            else:
                lane = None
                print(f"[WAY {way_id}] skip (no data)")
        
            if lane is None:
                print(f"[WAY {way_id}] skip (no valid lane)")
                continue

            print(f"[WAY {way_id}] → FINAL LANE = {lane}")

            lane_updates[int(float(way_id))] = lane
        # update osm
        osm_path = osm_obj.filepath
        output_osm = update_osm_lanes(osm_path, lane_updates)

        return output_osm
    def remove_row(evt: gr.SelectData, df):

        global fix_lanes

        if evt is None or evt.index is None:
            return df, df

        row, col = evt.index

        df = pd.DataFrame(df)

        if col == 2 and row < len(df):

            filename = df.iloc[row, 0]

            # ดึง image id จากชื่อไฟล์
            image_id = int(filename.split("_")[0])

            # ลบจาก global dict
            if image_id in fix_lanes:
                del fix_lanes[image_id]

            # ลบจากตาราง
            df = df.drop(row).reset_index(drop=True)

        return df.values.tolist(), df.values.tolist()
    # ---------- Events ----------

    upload_btn.click(
        upload_osm,
        inputs=file_input,
        outputs=[osm_state, error_modal, page_upload, page_bbox]
    )

    btn1.click(
        go_to_highway,
        inputs=[lon_min, lat_min, lon_max, lat_max],
        outputs=[bbox_state, map_view_scope, page_bbox, page_highway]
    )

    btn2.click(
        process_highway,
        inputs=highway,
        outputs=highway
    ).then(
        plot_highways,
        inputs=[osm_state, bbox_state, highway],
        outputs=[
            loading_overlay2,
            result_map,
            nodes_state,
            edges_state,
            page_highway,
            page_result
        ]
    ).then(
        extract_road_names,
        inputs=edges_state,
        outputs=road_name
    )

    btn3.click(
        go_to_page5,
        inputs=[edges_state, nodes_state, road_name, highway_type_input],
        outputs=[road_name_state,highway_type_state,final_map,result_info,filtered_edges_state,page_result, page_final]
    )

    btn4.click(
        download_streetview,
        inputs=[apikey_input, filtered_edges_state, road_name_state],
        outputs=[loading_overlay],
        show_progress=False
    ).then(
        lambda: render_loading(None, "Processing..."),
        None,
        loading_overlay
    ).then(
        run_lane_pipeline,
        inputs=[road_name_state, filtered_edges_state],
        outputs=csv_path_state
    ).then(
        load_images,
        inputs=road_name_state,
        outputs=[images_state, index_state, image_view, lane_state, lane_text]
    ).then(
        lambda imgs: f"Image : 1 / {len(imgs)}",
        inputs=images_state,
        outputs=counter
    ).then(
        show_page,
        None,
        [page_final, page_show]
    )
    back_to_filter_btn.click(
        back_to_filter,
        None,
        [page_final, page_result]
    )
    sidebar_btn.click(
        toggle_sidebar,
        inputs=sidebar_state,
        outputs=[sidebar_state, sidebar_panel, content_wrapper]
    )
    submit_btn.click(
        lambda: render_loading(None, "Updating file..."),
        None,
        status_text,
        show_progress=False
    ).then(
        submit_fix,
        inputs=[osm_state, csv_path_state],
        outputs=updated_osm_state
    ).then(
        lambda path: (
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            make_download_button(path)
        ),
        inputs=updated_osm_state,
        outputs=[status_text, page_show, page_download, download_button]
    )
    skip_btn.click(
        skip_image,
        inputs=[images_state, index_state, lane_state],
        outputs=[
            index_state,
            image_view,
            lane_text,
            counter,
            select_btn,
            skip_btn
        ]
    )
    select_btn.click(
        select_image,
        inputs=[selected_data, images_state, index_state, lane_state],
        outputs=[
            selected_data,
            selected_table,
            index_state,
            image_view,
            lane_text,
            counter,
            select_btn,
            skip_btn
        ]
    )
    selected_table.select(
        remove_row,
        inputs=selected_table,
        outputs=[selected_data, selected_table]
    )
    viewer_back_btn.click(
        go_back,
        inputs=[images_state, index_state, lane_state, selected_data],
        outputs=[
            index_state,
            image_view,
            lane_text,
            counter,
            selected_data,
            selected_table,
            select_btn,
            skip_btn
        ]
    )
demo.queue(concurrency_count=1).launch()