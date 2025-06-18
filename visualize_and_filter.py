import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from natsort import natsorted
from filterpy.kalman import KalmanFilter
import csv

# CONFIG
BASE_DIR = r"C:\Users\sudee\OneDrive\Documents\3D_ComputerVision\2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
EPS = 0.4
MIN_POINTS = 10
DIST_THRESHOLD = 1.5
DISPLAY = True
FRAME_DELAY = 0.1
OUTPUT_CSV = "tracked_objects.csv"

# TRACKER STATE
next_track_id = 0
trackers = {}  # track_id: {"kf": KalmanFilter, "last_update": int}


def create_kalman_filter(init_pos):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x[:3] = init_pos.reshape(3, 1)  # position
    kf.F = np.array([[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    kf.P *= 10
    kf.R = np.eye(3) * 0.5
    kf.Q = np.eye(6) * 0.01
    return kf


def read_velodyne_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def apply_filters(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(0.1)
    cropped = down.crop(o3d.geometry.AxisAlignedBoundingBox((-20, -10, -2), (30, 10, 2)))
    filtered, _ = cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return filtered


def detect_clusters(pcd):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=EPS, min_samples=MIN_POINTS).fit(points)
    labels = clustering.labels_

    centroids, boxes = [], []
    for cid in range(labels.max() + 1):
        mask = labels == cid
        cluster_pts = points[mask]
        if len(cluster_pts) < 30:
            continue
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        boxes.append(bbox)
        centroids.append(center)
    return centroids, boxes


def match_and_update(centroids, frame_id):
    global next_track_id, trackers
    assigned = {}
    cur_centroids = np.array(centroids)

    # Step 1: Prediction
    predicted_positions = []
    track_ids = list(trackers.keys())
    for tid in track_ids:
        trackers[tid]["kf"].predict()
        predicted_positions.append(trackers[tid]["kf"].x[:3].flatten())

    if len(predicted_positions) == 0:
        # First frame
        for c in cur_centroids:
            kf = create_kalman_filter(np.array(c))
            trackers[next_track_id] = {"kf": kf, "last_update": frame_id}
            assigned[next_track_id] = c
            next_track_id += 1
        return assigned

    D = cdist(np.array(predicted_positions), cur_centroids)
    used_tracks, used_centroids = set(), set()

    for i, row in enumerate(D):
        j = np.argmin(row)
        if row[j] < DIST_THRESHOLD and i not in used_tracks and j not in used_centroids:
            tid = track_ids[i]
            trackers[tid]["kf"].update(cur_centroids[j].reshape(3, 1))
            trackers[tid]["last_update"] = frame_id
            assigned[tid] = trackers[tid]["kf"].x[:3].flatten()
            used_tracks.add(i)
            used_centroids.add(j)

    # Step 2: New detections
    for j, c in enumerate(cur_centroids):
        if j not in used_centroids:
            kf = create_kalman_filter(c)
            trackers[next_track_id] = {"kf": kf, "last_update": frame_id}
            assigned[next_track_id] = c
            next_track_id += 1

    return assigned


def draw_scene(pcd, boxes, assignments):
    scene = [pcd]
    for tid, bbox in zip(assignments.keys(), boxes):
        bbox.color = (1, 0, 0)
        label = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        label.translate(bbox.get_center())
        scene.extend([bbox, label])
    return scene


def main():
    files = natsorted([f for f in os.listdir(BASE_DIR) if f.endswith(".bin")])
    vis = o3d.visualization.Visualizer() if DISPLAY else None
    if vis:
        vis.create_window("Tracking with Kalman Filter", width=960, height=720)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_id", "track_id", "x", "y", "z"])

        for frame_id, fname in enumerate(files):
            fpath = os.path.join(BASE_DIR, fname)
            print(f"📄 Frame {frame_id:03d}: {fname}")
            points = read_velodyne_bin(fpath)
            filtered_pcd = apply_filters(points)
            centroids, boxes = detect_clusters(filtered_pcd)
            assigned_tracks = match_and_update(centroids, frame_id)

            for tid, pos in assigned_tracks.items():
                writer.writerow([frame_id, tid, *pos])

            if DISPLAY:
                vis.clear_geometries()
                scene = draw_scene(filtered_pcd, boxes, assigned_tracks)
                for g in scene:
                    vis.add_geometry(g)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(FRAME_DELAY)

        if DISPLAY:
            vis.destroy_window()


if __name__ == "__main__":
    main()
