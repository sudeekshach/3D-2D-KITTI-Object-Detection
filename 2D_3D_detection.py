import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import open3d as o3d
from sklearn.cluster import DBSCAN

# === CONFIG ===
CALIB_PATH = r"C:\Users\sudee\OneDrive\Documents\3D_ComputerVision\2011_09_26_calib\2011_09_26"
VELO_PATH = r"C:\Users\sudee\OneDrive\Documents\3D_ComputerVision\2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
IMG_BASE_PATH = r"C:\Users\sudee\OneDrive\Documents\3D_ComputerVision\2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync"
YOLO_CSV = "detections_yolo.csv"
OUT_DIR = "projected_frames"
VIDEO_OUTPUT = "detections_output.mp4"

os.makedirs(OUT_DIR, exist_ok=True)

# === 1. Load Calibration ===
def load_calib(calib_path):
    def parse_line(line):
        key, value = line.split(':', 1)
        try:
            return key, np.array([float(x) for x in value.strip().split()])
        except ValueError:
            return key, value.strip()

    data = {}
    with open(os.path.join(calib_path, "calib_velo_to_cam.txt")) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = parse_line(line)
            if isinstance(val, np.ndarray):
                if val.size == 12:
                    data[key] = val.reshape(3, 4)
                elif val.size == 9:
                    data[key] = val.reshape(3, 3)

    Tr = data.get('Tr', np.hstack((data.get('R'), np.zeros((3, 1)))))

    data = {}
    with open(os.path.join(calib_path, "calib_cam_to_cam.txt")) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = parse_line(line)
            if isinstance(val, np.ndarray):
                data[key] = val

    R0 = data['R_rect_00'].reshape(3, 3)
    P2 = data['P_rect_02'].reshape(3, 4)
    return Tr, R0, P2

Tr_velo_to_cam, R0_rect, P2 = load_calib(CALIB_PATH)

# === 2. Load YOLO detections ===
yolo_df = pd.read_csv(YOLO_CSV)

# === 3. Projection ===
def project_lidar_to_image(pc_velo, Tr, R0, P):
    pc_velo = pc_velo[pc_velo[:, 0] > 0]  # only in front of car
    pts_hom = np.hstack((pc_velo[:, :3], np.ones((pc_velo.shape[0], 1))))  # Nx4
    pts_cam = (R0 @ (Tr @ pts_hom.T))  # 3xN
    pts_2d = P @ np.vstack((pts_cam, np.ones(pts_cam.shape[1])))  # 3xN
    pts_2d = pts_2d / pts_2d[2]
    return pts_2d[:2].T, pc_velo[:, :3]

# === 4. Match, Cluster and Draw ===
def match_and_draw(image, points_2d, points_3d, yolo_boxes, min_pts=10):
    for _, row in yolo_boxes.iterrows():
        cls, conf, x1, y1, x2, y2 = row['class'], row['confidence'], row['x1'], row['y1'], row['x2'], row['y2']
        box = (int(x1), int(y1), int(x2), int(y2))

        mask = [(x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2) for pt in points_2d]
        matched_pts_3d = points_3d[mask]

        if len(matched_pts_3d) < min_pts:
            continue  # Skip weak detections

        # Draw 2D box
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(image, f"{cls} {conf:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        for pt in matched_pts_3d:
            u, v = int(points_2d[np.where((points_3d == pt).all(axis=1))[0][0]][0]), int(points_2d[np.where((points_3d == pt).all(axis=1))[0][0]][1])
            cv2.circle(image, (u, v), 1, (255, 0, 0), -1)

    return image

print("\n🚀 Projecting and matching all frames...")
image_files = sorted([f for f in os.listdir(os.path.join(IMG_BASE_PATH, "image_02", "data")) if f.endswith('.png')])
frame_size = None
video_writer = None

for i, img_name in enumerate(tqdm(image_files)):
    base_name = img_name.split('.')[0]
    bin_file = os.path.join(VELO_PATH, f"{base_name}.bin")
    img_file = os.path.join(IMG_BASE_PATH, "image_02", "data", img_name)

    if not os.path.exists(bin_file):
        continue

    pc = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(img_file)
    if frame_size is None:
        frame_size = (img.shape[1], img.shape[0])
        video_writer = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 10, frame_size)

    pts_2d, pts_3d = project_lidar_to_image(pc, Tr_velo_to_cam, R0_rect, P2)
    frame_index = int(base_name)
    frame_dets = yolo_df[yolo_df['frame'] == frame_index]

    img_out = match_and_draw(img, pts_2d, pts_3d, frame_dets)
    cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}.png"), img_out)
    video_writer.write(img_out)

if video_writer:
    video_writer.release()

print(f"\n✅ Done. All projected frames saved in: {OUT_DIR} and video saved as: {VIDEO_OUTPUT}")
