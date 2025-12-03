import os
import cv2
import numpy as np
import torch
import sam2
from sam2.build_sam import build_sam2_video_predictor
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
checkpoint_path = "checkpoints/sam2_hiera_small.pt"
video_path = "data/IMG_2763.mp4"
frame_dir = "temp_frames"

# 1. Open Video to Detect iPhone FPS
temp_cap = cv2.VideoCapture(video_path)
fps = temp_cap.get(cv2.CAP_PROP_FPS)
temp_cap.release()

if fps == 0 or np.isnan(fps):
    print("Warning: Could not detect FPS. Defaulting to 30.")
    fps = 30.0

print(f"Detected Video FPS: {fps}")

# 2. Calculate Stride based on Mazzonetto et al. (2022)
# "One frame every 1.3 s was selected"
SECONDS_INTERVAL = .5
FRAME_STRIDE = int(fps * SECONDS_INTERVAL)
# FRAME_STRIDE = 6

# Safety check: ensure we don't stride 0 frames
if FRAME_STRIDE < 1: FRAME_STRIDE = 1

print(f"Strategy: Processing 1 frame every {SECONDS_INTERVAL}s (Stride = {FRAME_STRIDE} frames)")
# ==========================================

# 3. Setup SAM2 (Manual Config Load)
sam2_dir = os.path.dirname(sam2.__file__)
model_cfg = os.path.join(sam2_dir, "configs", "sam2", "sam2_hiera_s.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM2... (Device: {device})")
predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

# 4. Extract Frames with STRIDE
if os.path.exists(frame_dir):
    shutil.rmtree(frame_dir) # Automatically delete old 2000+ frames
os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_names = []
idx = 0
saved_count = 0

print(f"Extracting frames...")

# check if jpgs are avaliable already for the video

if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
    print(f"Frames already extracted in {frame_dir}. Skipping extraction.")
    frame_names = sorted(os.listdir(frame_dir))
    print(f"Found {len(frame_names)} frames.")
else: # only run if no frames are found
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if idx % FRAME_STRIDE == 0:
            # Resize iPhone 4K/1080p footage by 50% for speed
            # This aligns with the paper's finding that 1080p is sufficient
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            fname = f"{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(frame_dir, fname), frame)
            frame_names.append(fname)
            saved_count += 1
        idx += 1
    cap.release()

print(f"COMPLETED: Reduced {total_raw_frames} raw frames -> {len(frame_names)} frames for processing.")
inference_state = predictor.init_state(video_path=frame_dir)

# ==========================================
# 5. INTERACTIVE LANDMARK SELECTION
# ==========================================
def select_point_on_video(window_name, prompt):
    current_idx = 0
    selected_point = None
    selected_frame_idx = -1
    
    print(f"\n--- {prompt} ---")
    print("CONTROLS: [D] = Next Frame | [A] = Prev Frame | [CLICK] = Select Point | [ENTER] = Confirm")
    
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            print(f"Clicked at {x}, {y} on frame {current_idx}")

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_path = os.path.join(frame_dir, frame_names[current_idx])
        img = cv2.imread(img_path)
        
        cv2.putText(img, f"Frame: {current_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, prompt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if selected_point:
            cv2.circle(img, selected_point, 5, (0, 0, 255), -1)
            cv2.putText(img, "Selected! Press ENTER", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d'): # Next
            current_idx = min(current_idx + 1, len(frame_names) - 1)
            selected_point = None 
        elif key == ord('a'): # Prev
            current_idx = max(current_idx - 1, 0)
            selected_point = None
        elif key == 13: # Enter
            if selected_point is not None:
                selected_frame_idx = current_idx
                break
            else:
                print("Please click a point first!")
    
    cv2.destroyWindow(window_name)
    return selected_frame_idx, np.array(selected_point, dtype=np.float32)

# --- EXECUTE SELECTION ---
idx_nas, pt_nas = select_point_on_video("1/3 Select Nasion", "Find Frontal View -> Click NASION")
idx_lpa, pt_lpa = select_point_on_video("2/3 Select Left Ear", "Find Side View -> Click LEFT EAR (LPA)")
idx_rpa, pt_rpa = select_point_on_video("3/3 Select Right Ear", "Find Other Side -> Click RIGHT EAR (RPA)")

# Electrode Selection
print("\n--- ELECTRODE SELECTION ---")
print(f"Opening Frame {idx_nas} (Nasion view) for electrodes...")
img_path = os.path.join(frame_dir, frame_names[idx_nas])
img = cv2.imread(img_path)
electrode_points = []

def click_electrodes(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        electrode_points.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow("Select Electrodes", img)

cv2.imshow("Select Electrodes", img)
cv2.setMouseCallback("Select Electrodes", click_electrodes)
print("Click all visible electrodes. Press ENTER to finish.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================
# 6. TRACKING & SAVING
# ==========================================
print("\nInitializing Tracking...")

# Add Landmarks (Correctly mapped to their specific frames)
predictor.add_new_points_or_box(inference_state, frame_idx=idx_nas, obj_id=0, points=[pt_nas], labels=[1])
predictor.add_new_points_or_box(inference_state, frame_idx=idx_lpa, obj_id=1, points=[pt_lpa], labels=[1])
predictor.add_new_points_or_box(inference_state, frame_idx=idx_rpa, obj_id=2, points=[pt_rpa], labels=[1])

# Add Electrodes
current_id = 3
for pt in electrode_points:
    predictor.add_new_points_or_box(inference_state, frame_idx=idx_nas, obj_id=current_id, points=[np.array(pt, dtype=np.float32)], labels=[1])
    current_id += 1

print(f"Propagating... (Processing {len(frame_names)} frames)")
video_results = {}

for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    frame_data = {}
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = np.mean(xs), np.mean(ys)
            frame_data[out_obj_id] = (cx, cy)
    
    video_results[out_frame_idx] = frame_data
    if out_frame_idx % 10 == 0:
        print(f"Processed frame {out_frame_idx}/{len(frame_names)}")

import pickle
os.makedirs("results", exist_ok=True)
with open("results/tracking_data.pkl", "wb") as f:
    pickle.dump(video_results, f)

print(f"\nDONE! Saved tracking data for {len(video_results)} frames.")
print("NOTE: Coordinates are scaled by 0.5. Multiply by 2 for original resolution.")