โครงสร้างไฟล์ของโปรเจกต์ (Project Structure)
โปรเจกต์นี้เป็นระบบตรวจจับและแก้ไขจำนวนเลนถนน โดยใช้ภาพจาก Google Street View และข้อมูลจาก OpenStreetMap (OSM) โดยมีการจัดโครงสร้างไฟล์ตามหน้าที่ดังนี้:

1. core/

โฟลเดอร์สำหรับประมวลผลหลักของระบบ (Main Processing Logic)

pipeline.py : ควบคุมลำดับการทำงานของระบบทั้งหมด
model.py : เรียกใช้งานโมเดลตรวจจับเลน (LaneNet, UNet, DeepLabV3+)
count.py : คำนวณจำนวนเลนจากผลลัพธ์ของโมเดล
osm_logic.py : จัดการข้อมูลแผนที่ OSM และการจับคู่ข้อมูล
osm_update.py : อัปเดตจำนวนเลนกลับไปยังไฟล์ OSM

2. ui/

โฟลเดอร์สำหรับส่วนติดต่อผู้ใช้ (User Interface)

app.py : ไฟล์หลักสำหรับรันระบบด้วย Gradio
image_viewer.py : แสดงภาพ Street View และให้ผู้ใช้เลือกจำนวนเลน
map.py : แสดงแผนที่และพื้นที่ที่เลือก
loading.py : แสดงสถานะการโหลด

3. services/

โฟลเดอร์สำหรับเชื่อมต่อ API ภายนอก

streetview.py : ใช้เรียก Google Street View API เพื่อดึงภาพถนน

5. static/

โฟลเดอร์สำหรับไฟล์ตกแต่งหน้าจอ

style.css : กำหนดรูปแบบ UI

7. models/ 

โฟลเดอร์สำหรับเก็บโมเดล

lanenet/
lanenet-pytorch/
unet/
deeplab/

9. ไฟล์หลัก

Read.txt : อธิบายโปรเจกต์และโครงสร้างไฟล์

10.ไฟล์requirements

requirement_web.txt : ไลบรารีสำหรับ Web/Gradio
requirement_RTX4060_python3.12.3.txt : ไลบรารีสำหรับเครื่อง RTX4060
requirements_GTX1650_python3.7.txt : ไลบรารีสำหรับเครื่อง GTX1650
