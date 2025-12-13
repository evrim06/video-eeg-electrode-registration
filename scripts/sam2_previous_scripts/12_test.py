import sys
import os
import cv2
import numpy as np
import torch
import pickle
import json
import shutil
from tqdm import tqdm
from scipy.signal import savgol_filter
from ultralytics import YOLO
import sam2
from sam2.build_sam import build_sam2_video_predictor

torch.set_default_dtype(torch.float32)

# 1. CONFIGURATION & GLOBALS

# Device: CUDA if available, else CPU
if torch.cuda.is_available():
    DEVICE_STR = "cuda"
else:
    DEVICE_STR = "cpu"
print(f"--- Using device: {DEVICE_STR} ---")

# Paths (adjust VIDEO_PATH and YOLO_WEIGHTS as needed)
VIDEO_PATH   = "data/IMG_2763.mp4"
FRAME_DIR    = "frames"
RESULTS_DIR  = "results"
RAW_FILE     = os.path.join(RESULTS_DIR, "tracking_raw.pkl")
SMOOTH_FILE  = os.path.join(RESULTS_DIR, "tracking_smoothed.pkl")
ORDER_FILE   = os.path.join(RESULTS_DIR, "electrode_order.json")

# SAM2
SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG_NAME = "sam2_hiera_l.yaml"

SAM2_CONFIG = os.path.join(
    os.path.dirname(sam2.__file__),
    "configs",
    "sam2",
    SAM2_CONFIG_NAME
)

CONFIG = {
    "display_height": 800,      # GUI window height
    "duplicate_radius": 50,     # pixels (in cropped space) to treat as duplicates
    "yolo_conf": 0.25,          # YOLO confidence threshold
    "smooth_window": 7,         # Savitzky-Golay window length (odd)
    "poly_order": 2             # Savitzky-Golay polynomial order
}

YOLO_WEIGHTS = "runs/detect/train/weights/best_v2.pt"


# 2. HELPER FUNCTIONS

def initialize_models(
    yolo_weights_path=YOLO_WEIGHTS,
    sam2_cfg_path=SAM2_CONFIG,
    sam2_ckpt_path=SAM2_CHECKPOINT,
    device_used=DEVICE_STR
):
    print(f"Loading YOLO from {yolo_weights_path}...")

    if not os.path.exists(yolo_weights_path):
        print(f"ERROR: Custom YOLO weights not found at {yolo_weights_path}")
        sys.exit(1)

    yolo = YOLO(yolo_weights_path)

    sam2_predictor = build_sam2_video_predictor(
        sam2_cfg_path,
        sam2_ckpt_path,
        device=device_used
    )

    # --- FIX: Force all weights and buffers to float32 ---
    sam2_predictor.to(torch.float32)
    for p in sam2_predictor.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
    for b in sam2_predictor.buffers():
        b.data = b.data.float()

    return yolo, sam2_predictor


def resize_for_display(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height))
    return resized, scale


def draw_hud(img, idx, total, current_id):
    msg1 = f"Frame: {idx}/{total-1}"
    if current_id == 0:
        msg2 = "Click NAS (Nasion)"
    elif current_id == 1:
        msg2 = "Click LPA (Left Ear)"
    elif current_id == 2:
        msg2 = "Click RPA (Right Ear)"
    else:
        msg2 = f"Electrodes: ID start={current_id} | 'd': YOLO detect | Space: finish"

    cv2.putText(img, msg1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(img, msg2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, "Controls: s=forward, a=back, d=detect, space=finish, q=quit",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def extract_frames_with_crop():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        sys.exit(1)

    disp, scale = resize_for_display(first_frame, CONFIG["display_height"])
    print("Draw ROI around HEAD and press ENTER/SPACE.")
    roi = cv2.selectROI("Crop Selection", disp, fromCenter=False)
    cv2.destroyWindow("Crop Selection")

    sx = int(roi[0] / scale)
    sy = int(roi[1] / scale)
    sw = int(roi[2] / scale)
    sh = int(roi[3] / scale)

    if sw < 50 or sh < 50:
        print("Warning: ROI too small. Using full frame.")
        sx, sy = 0, 0
        sh, sw = first_frame.shape[0], first_frame.shape[1]

    offset_x, offset_y = sx, sy
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    idx = 0

    print("Extracting cropped frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[sy:sy+sh, sx:sx+sw]
        fname = f"{idx:05d}.jpg"
        cv2.imwrite(os.path.join(FRAME_DIR, fname), crop)
        frames.append(fname)
        idx += 1

    cap.release()
    print(f"Extracted {len(frames)} cropped frames.")
    return frames, (offset_x, offset_y)


def order_electrodes_head_relative(smoothed_data):
    relative_positions = {}

    print(f"Calculating head-relative positions across {len(smoothed_data)} frames...")

    for frame_idx, objects in smoothed_data.items():
        if 0 not in objects or 1 not in objects or 2 not in objects:
            continue
        if objects[0] is None or objects[1] is None or objects[2] is None:
            continue

        NAS = np.array(objects[0])
        LPA = np.array(objects[1])
        RPA = np.array(objects[2])

        ear_center = (LPA + RPA) / 2.0
        vec_x = RPA - LPA
        norm_x = np.linalg.norm(vec_x)
        if norm_x == 0: continue
        unit_x = vec_x / norm_x

        vec_y = NAS - ear_center
        norm_y = np.linalg.norm(vec_y)
        if norm_y == 0: continue
        unit_y = vec_y / norm_y

        for obj_id, coords in objects.items():
            if obj_id < 3: continue
            if coords is None: continue

            P = np.array(coords)
            v = P - ear_center
            px = np.dot(v, unit_x)
            py = np.dot(v, unit_y)

            relative_positions.setdefault(obj_id, []).append([px, py])

    final_stats = []
    for obj_id, rel_list in relative_positions.items():
        if len(rel_list) < 5: continue
        avg_rel = np.mean(rel_list, axis=0)
        final_stats.append({
            "id": obj_id,
            "rel_x": avg_rel[0],
            "rel_y": avg_rel[1]
        })

    sorted_electrodes = sorted(final_stats, key=lambda e: (-e["rel_y"], e["rel_x"]))
    return sorted_electrodes

def is_global_duplicate(candidate, existing_points, radius):
    for pt in existing_points:
        if np.linalg.norm(np.array(candidate) - np.array(pt)) < radius:
            return True
    return False


# 3. MAIN LOGIC

def main():
    yolo, sam2_predictor = initialize_models()
    frame_names, (crop_off_x, crop_off_y) = extract_frames_with_crop()

    if not frame_names:
        print("No frames extracted, aborting.")
        return

    state = sam2_predictor.init_state(video_path=FRAME_DIR)
    
    first_img = cv2.imread(os.path.join(FRAME_DIR, frame_names[0]))
    disp0, DISPLAY_SCALE = resize_for_display(first_img, CONFIG["display_height"])

    global_points = []
    current_id = 0
    current_idx = 0

    def on_click(event, x, y, flags, param):
        nonlocal current_id
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(x / DISPLAY_SCALE)
            real_y = int(y / DISPLAY_SCALE)
            real = (real_x, real_y)

            #FIX APPLIED HERE: Explicitly use Float32 for points 
            point_array = np.array([real], dtype=np.float32)

            sam2_predictor.add_new_points_or_box(
                state,
                frame_idx=current_idx,
                obj_id=current_id,
                points=point_array, 
                labels=np.array([1], dtype=np.int32)
            )
            global_points.append(real)
            current_id += 1

    cv2.namedWindow("Pipeline")
    cv2.setMouseCallback("Pipeline", on_click)

    print("\n--- Interactive Phase ---")
    print("Click NAS, then LPA, then RPA. Press 'd' to detect electrodes with YOLO.")

    while True:
        img_path = os.path.join(FRAME_DIR, frame_names[current_idx])
        img = cv2.imread(img_path)
        if img is None: break

        disp, _ = resize_for_display(img, CONFIG["display_height"])

        for i, pt in enumerate(global_points):
            color = (0, 255, 0) if i < 3 else (0, 0, 255)
            cv2.circle(disp, (int(pt[0]*DISPLAY_SCALE), int(pt[1]*DISPLAY_SCALE)), 5, color, -1)
            cv2.putText(disp, str(i), (int(pt[0]*DISPLAY_SCALE)+5, int(pt[1]*DISPLAY_SCALE)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        disp = draw_hud(disp, current_idx, len(frame_names), current_id)
        cv2.imshow("Pipeline", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            current_idx = min(current_idx + 15, len(frame_names)-1)
        elif key == ord('a'):
            current_idx = max(current_idx - 15, 0)
        elif key == ord('q'):
            sys.exit(0)
        elif key == ord('d'):
            if current_id < 3:
                print(">>> Please click NAS, LPA, RPA first!")
                continue

            print(f"Running YOLO on frame {current_idx}...")
            results = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
            found = 0

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    candidate = (cx, cy)
                    if is_global_duplicate(candidate, global_points, CONFIG["duplicate_radius"]):
                        continue
                    
                    # FIX APPLIED HERE: Pass box as float32
                    box_f32 = box.astype(np.float32)

                    sam2_predictor.add_new_points_or_box(
                        state,
                        frame_idx=current_idx,
                        obj_id=current_id,
                        box=box_f32
                    )
                    global_points.append(candidate)
                    current_id += 1
                    found += 1
            print(f"YOLO added {found} electrode(s).")

        elif key == 32:
            if current_id < 3:
                print(">>> Cannot finish. Please click NAS, LPA, RPA first.")
            else:
                print("Ending interactive phase.")
                break

    cv2.destroyAllWindows()

    # 6. Propagate with SAM2 (tracking)
    print("\n--- SAM2 Tracking ---")
    tracking = {}

    # --- FIX APPLIED HERE: Disable Autocast ---
    # Even if weights are float32, newer GPUs will try to autocast input to BFloat16.
    # We must explicitly disable this to ensure mat1 and mat2 are both float32.
    with torch.inference_mode(), torch.autocast(DEVICE_STR, enabled=False):
        for f_idx, ids, logits in tqdm(
            sam2_predictor.propagate_in_video(state),
            total=len(frame_names)
        ):
            frame_dict = {}
            for i, obj_id in enumerate(ids):
                mask = (logits[i] > 0.0).cpu().numpy().squeeze()
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx = np.mean(xs) + crop_off_x
                    cy = np.mean(ys) + crop_off_y
                    frame_dict[int(obj_id)] = (float(cx), float(cy))
            tracking[int(f_idx)] = frame_dict

            if f_idx % 100 == 0:
                with open(RAW_FILE, "wb") as f:
                    pickle.dump(tracking, f)

    with open(RAW_FILE, "wb") as f:
        pickle.dump(tracking, f)
    print(f"Raw tracking saved to: {RAW_FILE}")

    # 7. Smoothing
    print("\n--- Smoothing trajectories ---")
    smoothed = {}
    for f_idx, objs in tracking.items():
        smoothed[f_idx] = dict(objs)

    traj = {}
    for f_idx, objs in tracking.items():
        for obj_id, (x, y) in objs.items():
            traj.setdefault(obj_id, {"frames": [], "x": [], "y": []})
            traj[obj_id]["frames"].append(f_idx)
            traj[obj_id]["x"].append(x)
            traj[obj_id]["y"].append(y)

    for obj_id, t in traj.items():
        frames = t["frames"]
        xs = t["x"]
        ys = t["y"]
        if len(xs) >= CONFIG["smooth_window"]:
            try:
                sx = savgol_filter(xs, CONFIG["smooth_window"], CONFIG["poly_order"])
                sy = savgol_filter(ys, CONFIG["smooth_window"], CONFIG["poly_order"])
                for k, f_idx in enumerate(frames):
                    smoothed[f_idx][obj_id] = (float(sx[k]), float(sy[k]))
            except Exception:
                pass

    with open(SMOOTH_FILE, "wb") as f:
        pickle.dump(smoothed, f)
    print(f"Smoothed tracking saved to: {SMOOTH_FILE}")

    # 8. Rotation-invariant ordering
    print("\n--- Rotation-invariant electrode ordering ---")
    ordered = order_electrodes_head_relative(smoothed)

    if not ordered:
        print("Warning: Could not compute a robust order (maybe landmarks missing?).")
    else:
        print(f"{'Rank':<5} | {'ID':<5} | {'Front/Back (rel_y)':<18} | {'Left/Right (rel_x)':<18}")
        print("-" * 60)
        for i, e in enumerate(ordered):
            print(f"{i+1:<5} | {e['id']:<5} | {e['rel_y']:<18.2f} | {e['rel_x']:<18.2f}")

        ordered_ids = [e["id"] for e in ordered]
        with open(ORDER_FILE, "w") as f:
            json.dump(ordered_ids, f)
        print(f"\nElectrode order saved to: {ORDER_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()