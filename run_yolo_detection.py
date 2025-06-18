from ultralytics import YOLO
import os
import pandas as pd
import cv2

IMG_DIR = r"2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync\image_02\data"
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt for more accuracy

results = []
for file in sorted(os.listdir(IMG_DIR)):
    if not file.endswith(".png"):
        continue
    frame_id = int(file.split(".")[0])
    img_path = os.path.join(IMG_DIR, file)
    dets = model(img_path)[0].boxes.cpu().numpy()

    for box in dets:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        results.append({
            "frame": frame_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
            "class": cls
        })

pd.DataFrame(results).to_csv("detections_yolo.csv", index=False)
