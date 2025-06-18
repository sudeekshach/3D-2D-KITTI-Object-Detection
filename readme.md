# 🔍 3D + 2D Object Detection & Tracking on the KITTI Dataset

🚗 **Fusion‑based perception pipeline** that combines **YOLOv8 2‑D detections** with **3‑D LiDAR clustering**. The project projects LiDAR into camera space, matches 2‑D and 3‑D detections, overlays both on every KITTI image frame, and renders a full video.

---

## ✨ Features

|  ✔    |  Feature                                         |
| ----- | ------------------------------------------------ |
|  ✅    | YOLOv8 2‑D object detection                      |
|  ✅    | DBSCAN clustering of 3‑D point clouds            |
|  ✅    | LiDAR → camera projection via KITTI calibration  |
|  ✅    | 2‑D/3‑D matching & false‑positive filtering      |
|  ✅    | Frame export + MP4 video generation              |
|  🛠️  | Hooks for Kalman tracking, IoU matching, etc.    |

---

## 📁 Repository Layout

```text
.
├── data/
│   ├── 2011_09_26_calib/           # KITTI calibration files
│   ├── image_02/data/             # KITTI RGB frames (.png)
│   └── velodyne_points/data/      # KITTI LiDAR scans (.bin)
│
├── projected_frames/              # Auto‑generated overlays (after running)
├── detections_output.mp4          # Final video (auto‑generated)
│
├── run_yolo_on_images.py          # YOLO inference on all frames
├── 2D_3D_detection.py             # Fusion + video script (main)
├── detections_yolo.csv            # YOLO results (auto‑generated)
├── requirements.txt               # Python deps
└── README.md                      # This file
```

---

## ⚙️ Requirements

```bash
python3 ‑m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Key libraries**

```
ultralytics
opencv‑python
open3d
numpy
pandas
scikit‑learn
tqdm
```

---

## 🚀 Quick Start

\### 1️⃣ Run YOLOv8 on KITTI images

```bash
python run_yolo_on_images.py \
       --img-dir data/2011_09_26_drive_0001_sync/image_02/data \
       --model yolov8n.pt        # use s/m/l for higher accuracy
# ➜ outputs detections_yolo.csv
```

\### 2️⃣ Fuse 2‑D & 3‑D and build video

```bash
python 2D_3D_detection.py \
       --calib data/2011_09_26_calib/2011_09_26 \
       --velodyne data/…/velodyne_points/data \
       --images   data/…/image_02/data \
       --yolo     detections_yolo.csv
# ➜ overlays saved to projected_frames/ & detections_output.mp4
```

> **Tip:** change `--fps` inside the script to speed up or slow down playback.

---

## 🎥 Sample Frame



- Green = YOLO 2‑D box
- Blue dots = LiDAR points validated inside that box

---

## 🔄 Roadmap / Coming Soon

- Kalman filter tracking for persistent IDs
- True 3‑D bounding boxes projected into image view
- Confidence fusion (2‑D/3‑D voting)
- ROS bag export

---

## 📚 Dataset

- **KITTI Raw 2011‑09‑26 Drive 0001**
- 1392×512 stereo images @ 10 Hz
- 3‑D Velodyne scans (\~100 k pts / frame)

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Open3D](https://github.com/isl-org/Open3D)
- [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/)

---

## 📄 License

Apache 2.0 — free for research & education. Please cite appropriately if you use this code.

