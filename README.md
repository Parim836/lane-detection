# Lane Detection Using Deep Learning Techniques
เครื่องมือนี้ถูกพัฒนาขึ้นเพื่อปรับปรุงข้อมูลจำนวนเลนถนนใน OpenStreetMap (OSM) โดยประยุกต์ใช้ร่วมกับแบบจำลองการเรียนรู้เชิงลึก (Deep Learning)
โดยเครื่องมือนี้รวบรวมโมเดลทั้งหมด 4 ตัว ได้แก่ Lanenet, Lanenet-PyTorch, DeepLabV3+, และ UNet++ เพื่อตรวจจับเส้นถนนจากภาพถนนในประเทศไทย 
พร้อมส่วนติดต่อผู้ใช้แบบเว็บสำหรับดูผล ตรวจสอบ และแก้ไขเส้นจราจรในไฟล์ OSM ก่อนบันทึกไฟล์ใหม่
## Installation
#### 1.Clone โปรเจกต์จาก GitHub
```bash
git clone https://github.com/Parim836/lane-detection
cd lane-detection
```
#### 2.เตรียมสภาพแวดล้อม (Environment Setup)

สำหรับการติดตั้งและเตรียมสภาพแวดล้อมเพื่อให้โปรเจกต์ Lane Detection Using Deep Learning Techniques ทำงานได้อย่างครบถ้วน จำเป็นต้องติดตั้ง Dependencies จากไฟล์ Requirements ทั้ง 3 ตัว ที่ปรากฏในโครงสร้างไฟล์ของโปรเจกต์ **แนะนำให้ใช้ Environment (venv / conda) แยกกันแต่ละส่วน เพื่อป้องกัน Dependency ชนกัน**

- แต่ละ Requirements ถูกออกแบบมาสำหรับ Python และ GPU ต่างเวอร์ชันกัน ไม่ควรติดตั้งรวมใน Environment เดียว

- 2.1 สำหรับโมเดล Lanenet และ Lanenet-pytorch ต้องติดตั้ง:

```bash
conda create -n LaneNet python=3.7
conda activate LaneNet
pip install -r requirements_GTX1650_python3.7.txt
```
- 2.2 สำหรับโมเดล DeepLabV3+ และ UNet++ ต้องติดตั้ง:
```bash
conda create -n Segment python=3.12
conda activate Segment
pip install -r requirement_RTX4060_python3.12.3.txt
```
- 2.3 สำหรับส่วนติดต่อกับผู้ใช้ (User Interface)
เพื่อให้ส่วนติดต่อกับผู้ใช้ (UI) และการเชื่อมต่อส่วนต่างๆ สามารถทำงานได้ ต้องติดตั้ง: 
```bash
conda create -n tools python=3.10
conda activate tools
pip install -r requirement_web.txt
```
หลังจากติดตั้ง Dependency เรียบร้อยแล้ว ให้แก้ไขชื่อ Environment ที่ใช้สำหรับรันโมเดลในไฟล์ `model.py` ให้ตรงกับ Environment ที่ได้สร้างไว้
สำหรับ Environment ของส่วนติดต่อกับผู้ใช้ จะใช้สำหรับรันและทดสอบระบบหลักของโปรเจกต์ เช่น การเปิดหน้าเว็บและเรียกใช้งานโมเดล
