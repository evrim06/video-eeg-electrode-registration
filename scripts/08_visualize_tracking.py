import cv2
import pickle
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
video_path = "data/IMG_2763.mp4"
pkl_path = "results/tracking_data.pkl"
output_video_path = "results/debug_tracking.mp4"

# ==========================================
# 1. Load Data
# ==========================================
if not os.path.exists(pkl_path):
    print("Error: Could not find tracking_data.pkl")
    exit()

with open(pkl_path, "rb") as f:
    tracking_data = pickle.load(f)

print(f"Loaded tracking data for {len(tracking_data)} frames.")

# ==========================================
# 2. Setup Video Writer
# ==========================================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Remember: We processed at 50% scale (fx=0.5).
# We will create the debug video at that same 50% size to match the coordinates perfectly.
new_w, new_h = int(width * 0.5), int(height * 0.5)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 5.0, (new_w, new_h)) # 5 FPS for slow-motion review

print(f"Generating debug video: {output_video_path}...")

# ==========================================
# 3. Draw and Save
# ==========================================
frame_idx = 0
processed_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Only process frames that we actually tracked (the strided ones)
    if frame_idx in tracking_data:
        # Resize to match the tracking scale (50%)
        frame_small = cv2.resize(frame, (new_w, new_h), fx=0.5, fy=0.5)
        
        # Get points for this frame
        points_dict = tracking_data[frame_idx]
        
        # Draw Points
        for obj_id, coord in points_dict.items():
            if coord is None: continue # Skip lost points
            
            x, y = int(coord[0]), int(coord[1])
            
            # Color Coding:
            # ID 0 (Nasion), 1 (LPA), 2 (RPA) -> GREEN
            # ID 3+ (Electrodes) -> RED
            if obj_id <= 2:
                color = (0, 255, 0) # Green for Landmarks
                radius = 6
                # Add text label for landmarks
                label = ["Nas", "LPA", "RPA"][obj_id]
                cv2.putText(frame_small, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (0, 0, 255) # Red for Electrodes
                radius = 3
            
            cv2.circle(frame_small, (x, y), radius, color, -1)
        
        # Write frame to video
        out.write(frame_small)
        processed_count += 1
        
        if processed_count % 20 == 0:
            print(f"Rendered {processed_count} tracked frames...")

    frame_idx += 1

cap.release()
out.release()
print("Done! Open 'results/debug_tracking.mp4' to verify your tracking.")