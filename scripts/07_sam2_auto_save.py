import os
import cv2
import numpy as np
import torch
import sam2
from sam2.build_sam import build_sam2_video_predictor
import shutil
import pickle
from tqdm import tqdm # Progress bar


# CONFIGURATION

checkpoint_path = "checkpoints/sam2_hiera_small.pt"
video_path = "data/IMG_2763.mp4"
frame_dir = video_path.split("/")[-1].split(".")[0] + "_frames"
results_file = "results/tracking_data.pkl"

# Process every 6th frame
video_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
SECONDS_INTERVAL = 2
FRAME_STRIDE = int(video_fps * SECONDS_INTERVAL)


# 1. SETUP & FRAME EXTRACTION

sam2_dir = os.path.dirname(sam2.__file__)
model_cfg = os.path.join(sam2_dir, "configs", "sam2", "sam2_hiera_s.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM2 on {device}...")
predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

cap = cv2.VideoCapture(video_path)
frame_names = []
idx = 0
saved_count = 0

# check first if frame dir exists, only run extraction if it doesn't
if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 10:
    print(f"Frame directory {frame_dir} already exists. Skipping extraction.")
    # Populate frame_names with existing frames
    frame_names = sorted(os.listdir(frame_dir))
else:
    os.makedirs(frame_dir, exist_ok=True)
    print(f"Extracting frames (Stride={FRAME_STRIDE})...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
                break
        if idx % FRAME_STRIDE == 0:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # 50% scale
            fname = f"{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(frame_dir, fname), frame)
            frame_names.append(fname)
            saved_count += 1
        idx += 1
    cap.release()
    # If extracted, populate frame_names
    frame_names = sorted(os.listdir(frame_dir))

inference_state = predictor.init_state(video_path=frame_dir)

# 2. INTERACTIVE SELECTION

def select_point_on_video(window_name, prompt):
    current_idx = 0
    selected_point = None
    selected_frame_idx = -1
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_path = os.path.join(frame_dir, frame_names[current_idx])
        img = cv2.imread(img_path)
        cv2.putText(img, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "D=Next | A=Prev | Click | Enter", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if selected_point:
            cv2.circle(img, selected_point, 5, (0, 0, 255), -1)
            cv2.putText(img, "SELECTED", (selected_point[0]+10, selected_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d'): current_idx = min(current_idx + 1, len(frame_names) - 1)
        elif key == ord('a'): current_idx = max(current_idx - 1, 0)
        elif key == 13 and selected_point: # Enter
            selected_frame_idx = current_idx
            break
    
    cv2.destroyWindow(window_name)
    return selected_frame_idx, np.array(selected_point, dtype=np.float32)

idx_nas, pt_nas = select_point_on_video("Nasion", "Click Nasion (Front View)")
idx_lpa, pt_lpa = select_point_on_video("LPA", "Click Left Ear (Side View)")
idx_rpa, pt_rpa = select_point_on_video("RPA", "Click Right Ear (Side View)")

# Electrodes
print("\n--- ELECTRODE SELECTION ---")
img_path = os.path.join(frame_dir, frame_names[idx_nas])
img = cv2.imread(img_path)
electrode_points = []
def click_elec(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        electrode_points.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow("Electrodes", img)
cv2.imshow("Electrodes", img)
cv2.setMouseCallback("Electrodes", click_elec)
print("Click all visible electrodes. Press ENTER to finish.")
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3. TRACKING WITH AUTO-SAVE

print("\nInitializing Tracking...")
predictor.add_new_points_or_box(inference_state, frame_idx=idx_nas, obj_id=0, points=[pt_nas], labels=[1])
predictor.add_new_points_or_box(inference_state, frame_idx=idx_lpa, obj_id=1, points=[pt_lpa], labels=[1])
predictor.add_new_points_or_box(inference_state, frame_idx=idx_rpa, obj_id=2, points=[pt_rpa], labels=[1])

current_id = 3
for pt in electrode_points:
    predictor.add_new_points_or_box(inference_state, frame_idx=idx_nas, obj_id=current_id, points=[np.array(pt, dtype=np.float32)], labels=[1])
    current_id += 1

print(f"Propagating... (Results saved every 10 frames)")
video_results = {}
os.makedirs("results", exist_ok=True)

try:
    with torch.inference_mode():
        # Iterate through frames
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(predictor.propagate_in_video(inference_state), total=len(frame_names)):
            
            frame_data = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = np.mean(xs), np.mean(ys)
                    frame_data[out_obj_id] = (cx, cy)
            
            video_results[out_frame_idx] = frame_data
            
            # --- AUTO-SAVE FEATURE ---
            if out_frame_idx % 10 == 0:
                with open(results_file, "wb") as f:
                    pickle.dump(video_results, f)
                    
except KeyboardInterrupt:
    print("\n\n!!! SCRIPT STOPPED BY USER !!!")
    print("Saving what we have so far...")

# Final Save
with open(results_file, "wb") as f:
    pickle.dump(video_results, f)

print(f"\nDONE! Saved {len(video_results)} tracked frames to {results_file}")