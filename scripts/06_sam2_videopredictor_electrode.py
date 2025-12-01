import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sam2.build_sam import build_sam2_video_predictor

# ---------------------------------------------------------
# 1. Setup SAM2 Video Predictor
# ---------------------------------------------------------
checkpoint_path = "sam2_hiera_large.pt" 
model_cfg = "configs/sam2/sam2_hiera_l.yaml"                     
device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

# ---------------------------------------------------------
# 2. Load Video
# ---------------------------------------------------------
video_path = "data/IMG_2763.mp4"
# SAM2 requires the video frames to be extracted to a directory (usually)
# Or we can pass a list of frames. Let's extract frames to a folder for stability.
frame_dir = "temp_frames"
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_names = []
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # Save as JPEG (SAM2 reads from disk efficiently)
    fname = f"{count:05d}.jpg"
    cv2.imwrite(os.path.join(frame_dir, fname), frame)
    frame_names.append(fname)
    count += 1
cap.release()

# Initialize SAM2 state
inference_state = predictor.init_state(video_path=frame_dir)

# ---------------------------------------------------------
# 3. User Interaction: Click Landmarks & Electrodes (Frame 0)
# ---------------------------------------------------------
first_frame_path = os.path.join(frame_dir, frame_names[0])
first_frame = cv2.cvtColor(cv2.imread(first_frame_path), cv2.COLOR_BGR2RGB)

print("--- INSTRUCTIONS ---")
print("1. Click NASION, LEFT EAR, RIGHT EAR (Landmarks).")
print("2. Then click visible ELECTRODES.")
print("3. Press 'Enter' or close window when done.")

plt.figure(figsize=(10, 8))
plt.imshow(first_frame)
plt.title("Click: Landmarks first, then Electrodes")
points = plt.ginput(n=-1, timeout=0) # n=-1 means unlimited clicks
plt.close()

if not points:
    print("No points selected. Exiting.")
    exit()

points = np.array(points, dtype=np.float32)
# Labels: 1 means "positive click" (part of the object)
labels = np.ones(len(points), dtype=np.int32) 

# ---------------------------------------------------------
# 4. Add Points to SAM2 and Propagate
# ---------------------------------------------------------
print(f"Tracking {len(points)} objects...")

# We must give each click a unique Object ID.
# ID 0 = Nasion, ID 1 = LPA, ID 2 = RPA, ID 3+ = Electrodes
for obj_id, (pt, label) in enumerate(zip(points, labels)):
    
    # Add the click to Frame 0
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        points=[pt],
        labels=[label],
    )

# ---------------------------------------------------------
# 5. Propagate through the video (The Magic Step)
# ---------------------------------------------------------
# dictionary to store results: {frame_idx: {obj_id: (cx, cy)}}
video_results = {}

print("Propagating segmentation through video...")

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    
    frame_data = {}
    
    for i, out_obj_id in enumerate(out_obj_ids):
        # Extract the mask for this object
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        
        # Calculate Centroid of the mask
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = np.mean(xs), np.mean(ys)
            frame_data[out_obj_id] = (cx, cy)
        else:
            # Object might be occluded/turned away
            frame_data[out_obj_id] = None 

    video_results[out_frame_idx] = frame_data
    
    if out_frame_idx % 10 == 0:
        print(f"Processed frame {out_frame_idx}")

# ---------------------------------------------------------
# 6. Save Tracking Data for 3D Reconstruction
# ---------------------------------------------------------
# We filter out the None values (occlusions) later or save as NaN
import pickle
with open("results/tracking_data.pkl", "wb") as f:
    pickle.dump(video_results, f)

print("Tracking complete. Data saved.")