# Lane Detection Using Deep Learning Techniques
เครื่องมือนี้ถูกพัฒนาขึ้นเพื่อ **ปรับปรุงข้อมูลจำนวนเลนถนนใน OpenStreetMap (OSM)** โดยประยุกต์ใช้ร่วมกับแบบจำลองการเรียนรู้เชิงลึก (Deep Learning) ในการตรวจจับเส้นแบ่งเลนถนน

โดยเครื่องมือนี้มีการใช้งานร่วมกับแบบจำลองทั้งหมด 4 ตัว ได้แก่ LaneNet-TensoeFlow, LaneNet-PyTorch, DeepLabV3+, และ UNet++ เพื่อตรวจจับเส้นถนนจากภาพถนนในประเทศไทย 
พร้อมส่วนติดต่อผู้ใช้แบบเว็บสำหรับดูผล ตรวจสอบ และแก้ไขเส้นจราจรในไฟล์ OSM ก่อนบันทึกไฟล์ใหม่

#### แนวคิดของระบบ

ปัญหาหลักของ OpenStreetMap คือ **ข้อมูลจำนวนเลนถนน** อาจจะไม่ครบถ้วนหรือไม่ถูกต้อง เครื่องมือนี้จึงถูกพัฒนาขึ้นมาเพื่อช่วย:

- ตรวจจับเส้นจราจรจากภาพจริง
- ผู้ใช้ตรวจสอบก่อนแก้ไข
- ระบบนำผลลัพธ์มาเปรียบเที่ยบกับข้อมูลเดิมใน OpenStreetMap
- แก้ไขเฉพาะส่วนที่ต้องการ

## Installation
#### 1.Clone โปรเจกต์จาก GitHub
```bash
git clone https://github.com/Parim836/lane-detection
cd lane-detection
```
#### 2.เตรียมสภาพแวดล้อม (Environment Setup)

สำหรับการติดตั้งและเตรียมสภาพแวดล้อมเพื่อให้โปรเจกต์ Lane Detection Using Deep Learning Techniques ทำงานได้อย่างครบถ้วน จำเป็นต้องติดตั้ง Dependencies จากไฟล์ Requirements ทั้ง 3 ตัว ที่ปรากฏในโครงสร้างไฟล์ของโปรเจกต์ 
- แนะนำให้ใช้ Environment (venv / conda) แยกกันแต่ละส่วน เพื่อป้องกัน Dependency ชนกัน

- แต่ละ Requirements ถูกออกแบบมาสำหรับ Python และ GPU ต่างเวอร์ชันกัน ไม่ควรติดตั้งรวมใน Environment เดียว

2.1 สำหรับโมเดล LaneNet-TensorFlow และ LaneNet-PyTorch ต้องติดตั้ง:

```bash
conda create -n LaneNet python=3.7
conda activate LaneNet
pip install -r requirements_GTX1650_python3.7.txt
```
2.2 สำหรับโมเดล DeepLabV3+ และ UNet++ ต้องติดตั้ง:
```bash
conda create -n Segment python=3.12
conda activate Segment
pip install -r requirement_RTX4060_python3.12.3.txt
```
2.3 สำหรับส่วนติดต่อกับผู้ใช้ (User Interface)
เพื่อให้ส่วนติดต่อกับผู้ใช้ (UI) และการเชื่อมต่อส่วนต่างๆ สามารถทำงานได้ ต้องติดตั้ง: 
```bash
conda create -n tools python=3.10
conda activate tools
pip install -r requirement_web.txt
```
หลังจากติดตั้ง Dependency เรียบร้อยแล้ว ให้แก้ไข Path ของ Environment ที่ใช้สำหรับรันโมเดลในไฟล์ `model.py` ให้ตรงกับ Path ของ Environment ที่ได้สร้างไว้ โดยแก้ไขที่ `python_env` ของทุกโมเดลโดยวางให้ถูกต้อง
สำหรับในส่วนของ Environment ของส่วนติดต่อกับผู้ใช้ จะใช้สำหรับรันและทดสอบระบบหลักของโปรเจกต์ เช่น การเปิดหน้าเว็บและเรียกใช้งานโมเดล

## Test 
ก่อนจะทำการทดสอบระบบ ผู้ใช้จะต้องไปดาวน์โหลดไฟล์ checkpoint ของโมเดลก่อน เนื่องจากเป็นไฟล์ที่ผ่านการฝึกมาแล้ว ซึ่งใช้สำหรับการนำโมเดลไปประมวลผลภาพเพื่อทำ Lane Detection ได้อย่างถูกต้อง โดยสามารถดาวน์โหลด Checkpoint ได้จาก [Google Drive](https://drive.google.com/drive/folders/1nKJYvGWm7tRDAWfT4Hqeqs9ddIUrmamN?usp=sharing) 

#### การใช้งาน checkpoint
หลังจากดาวน์โหลดไฟล์เรียบร้อยแล้ว ให้ดำเนินการตามขั้นตอนของแต่ละโมเดลดังนี้:
1. โมเดล lanenet-lane-detection

   ให้นำไฟล์ Checkpoint ทั้งหมดที่ดาวน์โหลดมา ไปวางในโฟเดอร์ `model/lanenet-lane-detection/weights/tusimple_lanenet` จากนั้นทำการคัดลอกชื่อไฟล์ checkpoint (โดยไม่ต้องใส่นามสกุลไฟล์) แล้วทำการเปิดไฟล์ `model.py` และไปที่ฟังก์ชัน `LaneNet_TensorFlow` นำไปใส่ไว้ในตัวแปร `weight` 

3. โมเดล lanenet-lane-detection-pytorch

   ให้นำไฟล์ Checkpoint ทั้งหมดที่ดาวน์โหลดมา ไปวางในโฟเดอร์ร์ `model/lanenet-lane-detection-pytorch/log/checkpoint` โมเดลจะทำการโหลดไฟล์ Checkpoint จากตำแหน่งดังกล่าวโดยอัตโนมัติเมื่อเริ่มต้นการทำงาน
## Results

#### Lane Segmentation Results

|Model |IoU(%) | Dice(%)   |
|-------------|------------|------------|
| LaneNet     | 53.71 | 69.73    |
| LaneNet-PyTorch | 48.65 | 64.83 | 
| UNet++      | 51.80    | 66.48    | 
| DeepLabV3+  | 54.84    | 69.42    | 


#### Lane Detection Results

| Model        | Accuracy(%)   | Precision(%) | Recall(%) | F1-Score(%) |
|-------------|------------|------------|------------|------------|
| LaneNet     | 94.17 | 96.06    | 96.67 | 96.23 |
| LaneNet-PyTorch | 78.52 | 85.05 | 85.90 | 85.29 |
| UNet++      | 72.55    | 78.96    | 80.86 | 79.57 |
| DeepLabV3+  | 77.25    | 83.69    | 84.59 | 84.04 |

#### Combined Evaluation Results
เป็ยการวัดประสิทธิภาพของระบบหลังจากรวมผลจากหลายแบบจำลอง โดยจะประเมินว่า โมเดลสามารถจำแนกและจับเส้นเลนจากภาพถนนได้ถูกต้อง ครบถ้วน และแม่นยำแค่ไหน
##### Lane Segmentation Results
|Evaluation | Score(%)|
|-------------|------------|
| IoU(%)   | 48.64 | 
| Dice(%) | 65.32 | 


##### Lane Detection Results
|Evaluation | Score(%)|
|-------------|------------|
| Accuracy(%) | 89.25 |
| Precision(%)  | 93.17 |
|Recall(%)| 93.55 |
|  F1-Score(%)  | 93.08 |
