import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import shutil
import gc
import time
import urllib.request
from tqdm import tqdm
from scipy.signal import savgol_filter
from ultralytics import YOLO
from contextlib import nullcontext


# STEP 1: APPLY SAM2 DTYPE PATCH BEFORE IMPORTING SAM2

#Remove this patch once SAM2 upstream fixes dtype handling

print("=" * 70)
print("APPLYING SAM2 DTYPE PATCH v2...")
print("=" * 70)

torch.set_default_dtype(torch.float32)

import sam2
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from sam2.modeling.sam.transformer import RoPEAttention

# Store original forward methods
_original_rope_attention_forward = RoPEAttention.forward
_original_memory_attention_layer_forward = MemoryAttentionLayer.forward
_original_memory_attention_forward = MemoryAttention.forward


def _convert_to_dtype(obj, target_dtype):
    """
    Recursively convert tensors to target dtype.
    Handles: tensors, lists, tuples, and None.
    """
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype != target_dtype:
            return obj.to(target_dtype)
        return obj
    elif isinstance(obj, list):
        return [_convert_to_dtype(item, target_dtype) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_dtype(item, target_dtype) for item in obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_dtype(v, target_dtype) for k, v in obj.items()}
    else:
        return obj


def _get_model_dtype(module):
    """Get the dtype of the module's parameters."""
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _patched_rope_attention_forward(self, q, k, v, num_k_exclude_rope=0):
    """
    Patched RoPEAttention.forward that ensures all tensors have matching dtype.
    """
    target_dtype = _get_model_dtype(self)
    
    q = _convert_to_dtype(q, target_dtype)
    k = _convert_to_dtype(k, target_dtype)
    v = _convert_to_dtype(v, target_dtype)
    
    return _original_rope_attention_forward(self, q, k, v, num_k_exclude_rope)


def _patched_memory_attention_layer_forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
    """
    Patched MemoryAttentionLayer.forward that ensures dtype consistency.
    """
    target_dtype = _get_model_dtype(self)
    
    tgt = _convert_to_dtype(tgt, target_dtype)
    memory = _convert_to_dtype(memory, target_dtype)
    pos = _convert_to_dtype(pos, target_dtype)
    query_pos = _convert_to_dtype(query_pos, target_dtype)
    
    return _original_memory_attention_layer_forward(self, tgt, memory, pos, query_pos, num_k_exclude_rope)


def _patched_memory_attention_forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
    """
    Patched MemoryAttention.forward that ensures dtype consistency.
    Handles both tensors and lists of tensors.
    """
    target_dtype = _get_model_dtype(self)
    
    curr = _convert_to_dtype(curr, target_dtype)
    memory = _convert_to_dtype(memory, target_dtype)
    curr_pos = _convert_to_dtype(curr_pos, target_dtype)
    memory_pos = _convert_to_dtype(memory_pos, target_dtype)
    
    return _original_memory_attention_forward(self, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens)


# Apply patches
RoPEAttention.forward = _patched_rope_attention_forward
MemoryAttentionLayer.forward = _patched_memory_attention_layer_forward
MemoryAttention.forward = _patched_memory_attention_forward

print("✓ SAM2 dtype patch v2 applied!")

# Now import rest of SAM2
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# STEP 2: CONFIGURATION


if torch.cuda.is_available():
    DEVICE_STR = "cuda"
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Device: CUDA")
else:
    DEVICE_STR = "cpu"
    DEVICE = torch.device("cpu")
    print(f"Device: CPU")

print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

print(f"Project Root: {BASE_DIR}")

VIDEO_PATH      = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
FRAME_DIR       = os.path.join(BASE_DIR, "frames")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")

RAW_FILE        = os.path.join(RESULTS_DIR, "tracking_raw.pkl")
SMOOTH_FILE     = os.path.join(RESULTS_DIR, "tracking_smoothed.pkl")
ORDER_FILE      = os.path.join(RESULTS_DIR, "electrode_order.json")
CROP_INFO_FILE  = os.path.join(RESULTS_DIR, "crop_info.json")

YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")

SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2_hiera_small.pt")
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"

if not os.path.exists(SAM2_CHECKPOINT):
    print(f"Downloading SAM2 checkpoint...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, SAM2_CHECKPOINT)

SAM2_CONFIG_NAME = "sam2_hiera_s.yaml"
SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", SAM2_CONFIG_NAME)

if not os.path.exists(SAM2_CONFIG):
    for p in [os.path.join(BASE_DIR, "configs", "sam2", SAM2_CONFIG_NAME),
              os.path.join(BASE_DIR, "configs", SAM2_CONFIG_NAME)]:
        if os.path.exists(p):
            SAM2_CONFIG = p
            break

print(f"SAM2 Config: {SAM2_CONFIG}")

CONFIG = {
    "frame_skip": 10,  # less skipped frames = better detection          
    "flash_duration": 3.0,      
    "display_height": 800,      
    "duplicate_radius": 50,     
    "yolo_conf": 0.25,          
    "smooth_window": 7,         
    "poly_order": 2,            
    "cap_mask_expansion": 1.10,   
    "cap_min_area_frac": 0.03,    
    "cap_max_area_frac": 0.70,    
    "cap_confirm_alpha": 0.45,    
    "cap_autodetect_use_yolo": True, 
}

# STEP 3: MODEL INITIALIZATION


def initialize_models():
    print(f"\nLoading YOLO from {YOLO_WEIGHTS}...")
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"ERROR: YOLO weights not found")
        sys.exit(1)
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading SAM2 Video Predictor...")
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE_STR)
    
    # Convert to Float32
    sam2_predictor = sam2_predictor.float()
    sam2_predictor.eval()
    
    print("✓ Models initialized\n")
    return yolo, sam2_predictor


def build_sam2_float32():
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE_STR)
    model = model.float()
    model.eval()
    return model



# STEP 4: HELPER FUNCTIONS


def resize_for_display(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    return cv2.resize(img, (int(w * scale), target_height)), scale


def draw_hud(img, idx, total, current_id):
    msg1 = f"Frame: {idx}/{total-1}"
    status = [f"NAS: {'[Saved]' if current_id > 0 else '[Click]'}",
              f"LPA: {'[Saved]' if current_id > 1 else '[Click]'}",
              f"RPA: {'[Saved]' if current_id > 2 else '[Click]'}"]

    if current_id <= 2:
        msg2 = " -> ".join(status)
        color = (0, 255, 255)
    else:
        msg2 = f"Landmarks: [Saved] | Electrodes: {current_id - 3}"
        color = (0, 255, 0)

    cv2.rectangle(img, (0, 0), (img.shape[1], 130), (0, 0, 0), -1)
    cv2.putText(img, msg1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, msg2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, "NAV: [S] Fwd (+15) | [A] Back (-15) | [Q] Quit", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img, "ACT: [D] YOLO Detect | [Click] Add | [Space] Done", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return img


def interactive_crop(video_path, display_height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, roi, drawing, start = 0, None, False, None

    def mouse(event, x, y, flags, param):
        nonlocal roi, drawing, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, start, roi = True, (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi = (min(start[0], x), min(start[1], y), abs(x - start[0]), abs(y - start[1]))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Crop")
    cv2.setMouseCallback("Crop", mouse)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break

        disp, scale = resize_for_display(frame, display_height)
        show = disp.copy()
        cv2.rectangle(show, (0, 0), (show.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(show, "CROP: Draw box around cap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(show, "A/S: Move frames | SPACE: Confirm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        if roi:
            cv2.rectangle(show, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 255), 2)

        cv2.imshow("Crop", show)
        k = cv2.waitKey(30) & 0xFF

        if k == ord('a'): idx = max(0, idx - 15)
        elif k == ord('s'): idx = min(total - 1, idx + 15)
        elif k in (13, 32) and roi: break
        elif k == ord('q'): sys.exit(0)

    cv2.destroyWindow("Crop")
    cap.release()

    return int(roi[0]/scale), int(roi[1]/scale), int(roi[2]/scale), int(roi[3]/scale)


def extract_frames():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n--- Cropping ---")
    sx, sy, sw, sh = interactive_crop(VIDEO_PATH, CONFIG["display_height"])

    if sw < 50 or sh < 50:
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        sx, sy, sw, sh = 0, 0, frame.shape[1], frame.shape[0]

    cap = cv2.VideoCapture(VIDEO_PATH)
    skip = CONFIG["frame_skip"]
    frames, idx, count = [], 0, 0

    print(f"Extracting frames (skip={skip})...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % skip == 0:
            crop = frame[sy:sy+sh, sx:sx+sw]
            cv2.imwrite(os.path.join(FRAME_DIR, f"{idx:05d}.jpg"), crop)
            frames.append(f"{idx:05d}.jpg")
            idx += 1
        count += 1
    cap.release()

    with open(CROP_INFO_FILE, "w") as f:
        json.dump({"x": sx, "y": sy, "w": sw, "h": sh, "skip": skip}, f)

    print(f"✓ Extracted {len(frames)} frames")
    return frames, (sx, sy)


def order_electrodes(smoothed):
    positions = {}
    for fidx, objs in smoothed.items():
        if not all(i in objs and objs[i] for i in [0, 1, 2]):
            continue
        NAS, LPA, RPA = np.array(objs[0]), np.array(objs[1]), np.array(objs[2])
        center = (LPA + RPA) / 2
        ux = (RPA - LPA) / (np.linalg.norm(RPA - LPA) + 1e-8)
        uy = (NAS - center) / (np.linalg.norm(NAS - center) + 1e-8)
        for oid, coords in objs.items():
            if oid < 3 or coords is None: continue
            v = np.array(coords) - center
            positions.setdefault(oid, []).append([np.dot(v, ux), np.dot(v, uy)])

    stats = [{"id": oid, "y": np.mean(pts, axis=0)[1], "x": np.mean(pts, axis=0)[0]}
             for oid, pts in positions.items() if len(pts) >= 5]
    return sorted(stats, key=lambda e: (-e["y"], e["x"]))


def is_duplicate(pt, existing, radius):
    return any(np.linalg.norm(np.array(pt) - np.array(p)) < radius for p in existing[3:])



# STEP 5: CAP MASK FUNCTIONS


def expand_mask(mask, expansion=1.10):
    if expansion <= 1.0: return mask.astype(bool)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return mask.astype(bool)
    k = max(3, int(max(ys.ptp(), xs.ptp()) * (expansion - 1) * 0.5))
    if k % 2 == 0: k += 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def draw_mask_overlay(disp, mask, alpha=0.45):
    out = disp.copy()
    m = cv2.resize(mask.astype(np.uint8), (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = out.copy()
    overlay[m == 0] = (overlay[m == 0] * 0.35).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def manual_cap_mask(img, sam2, state, fidx, display_h):
    """User clicks center of cap"""
    disp, scale = resize_for_display(img, display_h)
    clicks = []

    def cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x/scale), int(y/scale)))

    win = "Click CENTER of cap | q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while not clicks:
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): sys.exit(0)
    cv2.destroyWindow(win)

    with torch.inference_mode():
        _, _, logits = sam2.add_new_points_or_box(
            state, frame_idx=fidx, obj_id=999,
            points=[np.array(clicks[0], dtype=np.float32)], labels=[1]
        )
    return (logits[0] > 0).cpu().numpy().squeeze().astype(bool)


def auto_cap_mask(img, yolo=None):
    """Automatic cap mask using AMG"""
    print("[Auto-Mask] Generating...")
    model = build_sam2_float32()
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.88,
                                      stability_score_thresh=0.92, min_mask_region_area=500)
    masks = amg.generate(img)
    del amg, model
    gc.collect()
    if DEVICE_STR == "cuda": torch.cuda.empty_cache()

    H, W = img.shape[:2]
    amin, amax = H*W*CONFIG["cap_min_area_frac"], H*W*CONFIG["cap_max_area_frac"]

    centers = []
    if CONFIG["cap_autodetect_use_yolo"] and yolo:
        res = yolo.predict(img, conf=0.2, verbose=False)
        if res and res[0].boxes is not None:
            centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in res[0].boxes.xyxy.cpu().numpy()]

    best, score = None, -1e9
    for m in masks:
        area = float(m["segmentation"].sum())
        if not (amin <= area <= amax): continue
        s = area + sum(1000 for cx, cy in centers if 0 <= int(cy) < H and 0 <= int(cx) < W and m["segmentation"][int(cy), int(cx)])
        if s > score: best, score = m["segmentation"].astype(bool), s

    return best if best is not None else (max(masks, key=lambda x: x["area"])["segmentation"].astype(bool) if masks else np.ones((H, W), bool))


def create_cap_mask(img, sam2, state, yolo):
    """Full UX for cap mask - defaults to manual"""
    mode = "manual"

    while True:
        if mode == "auto":
            mask = auto_cap_mask(img, yolo)
        else:
            mask = manual_cap_mask(img, sam2, state, 0, CONFIG["display_height"])

        mask_exp = expand_mask(mask, CONFIG["cap_mask_expansion"])
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        preview = draw_mask_overlay(disp, mask_exp, CONFIG["cap_confirm_alpha"])

        win = "Confirm Cap Mask"
        cv2.namedWindow(win)
        while True:
            show = preview.copy()
            cv2.rectangle(show, (0, 0), (show.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(show, f"Mode={mode} | y=OK | r=redo | m=manual | a=auto | q=quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(show, "Tip: Press 'm' and click center of cap for best results", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.imshow(win, show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('y'):
                with torch.inference_mode():
                    sam2.add_new_mask(state, frame_idx=0, obj_id=999, mask=mask_exp.astype(np.uint8))
                cv2.destroyWindow(win)
                return mask_exp
            if k == ord('r'): break
            if k == ord('m'): mode = "manual"; break
            if k == ord('a'): mode = "auto"; break
            if k == ord('q'): sys.exit(0)
        cv2.destroyWindow(win)


def precompute_cap_masks(sam2, state, frames):
    """Pre-track cap mask"""
    print("Pre-tracking cap mask...")
    masks = {}
    with torch.inference_mode():
        for fidx, ids, logits in tqdm(sam2.propagate_in_video(state), total=len(frames)):
            if 999 in ids:
                m = (logits[list(ids).index(999)] > 0).cpu().numpy().squeeze()
                masks[fidx] = expand_mask(m, CONFIG["cap_mask_expansion"])
            else:
                masks[fidx] = None
    return masks



# STEP 6: MAIN


def main():
    print("\n" + "=" * 70)
    print("EEG ELECTRODE REGISTRATION PIPELINE")
    print("(with SAM2 dtype patch v2)")
    print("=" * 70 + "\n")

    # 1. Models
    yolo, sam2 = initialize_models()

    # 2. Frames
    frames, (off_x, off_y) = extract_frames()
    if not frames: return

    # 3. Init state
    print("\nInitializing SAM2 state...")
    with torch.inference_mode():
        state = sam2.init_state(video_path=FRAME_DIR)

    # 4. Cap mask
    first_img = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    cap_mask = create_cap_mask(first_img, sam2, state, yolo)

    # 5. Pre-compute cap masks
    print("\nPre-computing cap masks...")
    with torch.inference_mode():
        cap_state = sam2.init_state(video_path=FRAME_DIR)
        sam2.add_new_mask(cap_state, frame_idx=0, obj_id=999, mask=cap_mask.astype(np.uint8))
    cap_cache = precompute_cap_masks(sam2, cap_state, frames)
    del cap_state
    gc.collect()
    if DEVICE_STR == "cuda": torch.cuda.empty_cache()

    # 6. Interactive labeling
    _, SCALE = resize_for_display(first_img, CONFIG["display_height"])
    points, current_id, idx_box, flashes = [], 0, [0], []

    def click(e, x, y, f, p):
        nonlocal current_id
        if e != cv2.EVENT_LBUTTONDOWN: return
        cidx = idx_box[0]
        rx, ry = int(x/SCALE), int(y/SCALE)

        cap = cap_cache.get(cidx)
        if current_id >= 3 and (cap is None or not (0 <= ry < cap.shape[0] and 0 <= rx < cap.shape[1] and cap[ry, rx])):
            print(">>> Outside cap mask")
            return

        with torch.inference_mode():
            sam2.add_new_points_or_box(state, frame_idx=cidx, obj_id=current_id,
                                       points=[np.array((rx, ry), dtype=np.float32)], labels=[1])
        points.append((rx, ry))
        flashes.append((rx, ry, (0, 255, 0) if current_id < 3 else (0, 0, 255), time.time()))
        current_id += 1

    cv2.namedWindow("Pipeline")
    cv2.setMouseCallback("Pipeline", click)

    print("\n--- Interactive Phase ---")
    while True:
        cidx = idx_box[0]
        img = cv2.imread(os.path.join(FRAME_DIR, frames[cidx]))
        if img is None: break
        disp, _ = resize_for_display(img, CONFIG["display_height"])

        cap = cap_cache.get(cidx)
        if cap is not None:
            overlay = disp.copy()
            rm = cv2.resize(cap.astype(np.uint8), (disp.shape[1], disp.shape[0]))
            overlay[rm == 0] = overlay[rm == 0] // 2
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)

        now = time.time()
        flashes[:] = [f for f in flashes if now - f[3] < CONFIG["flash_duration"]]
        for fx, fy, col, _ in flashes:
            cv2.circle(disp, (int(fx*SCALE), int(fy*SCALE)), 7, col, -1)

        disp = draw_hud(disp, cidx, len(frames), current_id)
        cv2.imshow("Pipeline", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'): idx_box[0] = min(cidx + 15, len(frames) - 1)
        elif k == ord('a'): idx_box[0] = max(cidx - 15, 0)
        elif k == ord('q'): sys.exit(0)
        elif k == ord('d'):
            if current_id < 3:
                print(">>> Click landmarks first!")
                continue
            print(f"YOLO detecting frame {cidx}...")
            res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
            found = 0
            if res and res[0].boxes is not None:
                cap = cap_cache.get(cidx)
                for box in res[0].boxes.xyxy.cpu().numpy():
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if cap is not None and not (0 <= int(cy) < cap.shape[0] and 0 <= int(cx) < cap.shape[1] and cap[int(cy), int(cx)]):
                        continue
                    if is_duplicate((cx, cy), points, CONFIG["duplicate_radius"]):
                        continue
                    with torch.inference_mode():
                        sam2.add_new_points_or_box(state, frame_idx=cidx, obj_id=current_id, box=box)
                    points.append((cx, cy))
                    flashes.append((cx, cy, (0, 0, 255), time.time()))
                    current_id += 1
                    found += 1
            print(f"Added {found} electrodes")
        elif k == 32:
            if current_id < 3:
                print(">>> Click landmarks first")
                continue
            break
    cv2.destroyAllWindows()

    # 7. TRACKING
    print("\n--- SAM2 Tracking ---")
    tracking = {}
    with torch.inference_mode():
        for fidx, ids, logits in tqdm(sam2.propagate_in_video(state), total=len(frames)):
            fd = {}
            for i, oid in enumerate(ids):
                if oid == 999: continue
                m = (logits[i] > 0).cpu().numpy().squeeze()
                ys, xs = np.where(m)
                if len(xs) > 0:
                    fd[int(oid)] = (float(np.mean(xs) + off_x), float(np.mean(ys) + off_y))
            tracking[int(fidx)] = fd

    # Save
    print("\nSaving results...")
    with open(RAW_FILE, "wb") as f: pickle.dump(tracking, f)

    # Smooth
    smoothed = {fidx: dict(objs) for fidx, objs in tracking.items()}
    traj = {}
    for fidx, objs in tracking.items():
        for oid, (x, y) in objs.items():
            traj.setdefault(oid, {"x": [], "y": [], "f": []})
            traj[oid]["f"].append(fidx)
            traj[oid]["x"].append(x)
            traj[oid]["y"].append(y)

    for oid, t in traj.items():
        if len(t["x"]) >= CONFIG["smooth_window"]:
            try:
                sx = savgol_filter(t["x"], CONFIG["smooth_window"], CONFIG["poly_order"])
                sy = savgol_filter(t["y"], CONFIG["smooth_window"], CONFIG["poly_order"])
                for k, fidx in enumerate(t["f"]):
                    smoothed[fidx][oid] = (float(sx[k]), float(sy[k]))
            except: pass

    with open(SMOOTH_FILE, "wb") as f: pickle.dump(smoothed, f)

    ordered = order_electrodes(smoothed)
    if ordered:
        with open(ORDER_FILE, "w") as f: json.dump([e["id"] for e in ordered], f)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults: {RAW_FILE}, {SMOOTH_FILE}, {ORDER_FILE}\n")


if __name__ == "__main__":
    main()