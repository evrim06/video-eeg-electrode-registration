import cv2
import numpy as np
import pickle
import os
import sys
import json
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration"
VIDEO_PATH = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
TRACKING_FILE = os.path.join(BASE_DIR, "results", "tracking_results.pkl")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "results", "final_tracking_overlay.mp4")
CROP_INFO_FILE = os.path.join(BASE_DIR, "results", "crop_info.json")

# --- LOAD DATA ---
if not os.path.exists(TRACKING_FILE):
    print("Error: Tracking file not found.")
    sys.exit(1)

with open(TRACKING_FILE, "rb") as f:
    tracking_data = pickle.load(f)

# Load Crop Info
if os.path.exists(CROP_INFO_FILE):
    with open(CROP_INFO_FILE, "r") as f:
        crop_info = json.load(f)
        OFFSET_X = crop_info.get("x", 0)
        OFFSET_Y = crop_info.get("y", 0)
        FRAME_SKIP = crop_info.get("skip", 1)
        print(f"Loaded Crop Info: Offset ({OFFSET_X}, {OFFSET_Y}), Skip {FRAME_SKIP}")
else:
    print("Warning: Crop info not found. Assuming full frame (0,0).")
    OFFSET_X, OFFSET_Y, FRAME_SKIP = 0, 0, 1

# --- SETUP VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# Adjust FPS: The output video will be shorter/faster because we skipped frames
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"Rendering video to: {OUTPUT_VIDEO}")

# --- PROCESS FRAMES ---
sorted_tracked_frames = sorted(tracking_data.keys())

# Color map
np.random.seed(42)
colors = {}

pbar = tqdm(total=len(sorted_tracked_frames))

for frame_idx in sorted_tracked_frames:
    # 1. Calculate the real video frame number
    # Script 1 extracts every Nth frame. frame_idx 0 = video 0, frame_idx 1 = video 1*SKIP
    target_video_frame = frame_idx * FRAME_SKIP
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_frame)
    ret, frame = cap.read()
    if not ret: break

    # 2. Draw Electrodes
    objs = tracking_data[frame_idx]
    
    # Draw Landmarks (ID 0, 1, 2)
    landmarks = [0, 1, 2]
    landmark_labels = ["NAS", "LPA", "RPA"]
    
    for i, lid in enumerate(landmarks):
        if lid in objs and objs[lid] is not None:
            cx, cy = objs[lid]
            
            # --- THE FIX IS HERE ---
            # Add the crop offset to map back to global video coordinates
            gx = int(cx + OFFSET_X)
            gy = int(cy + OFFSET_Y)
            # -----------------------
            
            cv2.circle(frame, (gx, gy), 8, (0, 0, 255), -1) 
            cv2.putText(frame, landmark_labels[i], (gx+10, gy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw Electrodes (ID 3+)
    for obj_id, coords in objs.items():
        if obj_id < 3 or coords is None: continue
        
        if obj_id not in colors:
            colors[obj_id] = np.random.randint(0, 255, 3).tolist()
        
        cx, cy = coords
        
        # --- THE FIX IS HERE ---
        gx = int(cx + OFFSET_X)
        gy = int(cy + OFFSET_Y)
        # -----------------------
        
        cv2.circle(frame, (gx, gy), 5, colors[obj_id], -1)
        cv2.putText(frame, str(obj_id), (gx+8, gy-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 3. Write Frame
    out.write(frame)
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print("\nDone! Open the video to verify tracking.")
print(f"File: {OUTPUT_VIDEO}")