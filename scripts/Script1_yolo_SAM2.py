"""
EEG Electrode Registration Pipeline 

Features:
1. YOLO detection + SAM2 tracking
2. Frame alignment using landmarks (NAS, LPA, RPA)
3. Head-centered coordinate normalization
4. Multi-frame averaging for stable positions

Output:
- tracking_results.pkl: Raw tracking per frame
- aligned_positions.json: Final averaged 2D positions (head-aligned)
- crop_info.json: Crop metadata
"""

import sys
import os
import cv2
import numpy as np
import torch
import pickle
import json
import shutil
import gc
import time
import urllib.request
from tqdm import tqdm
from ultralytics import YOLO
from scipy.spatial import procrustes



# STEP 1: SAM2 DTYPE PATCH


print("=" * 70)
print("APPLYING SAM2 DTYPE PATCH...")
print("=" * 70)

torch.set_default_dtype(torch.float32)

import sam2
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from sam2.modeling.sam.transformer import RoPEAttention

_original_rope_attention_forward = RoPEAttention.forward
_original_memory_attention_layer_forward = MemoryAttentionLayer.forward
_original_memory_attention_forward = MemoryAttention.forward

def _convert_to_dtype(obj, target_dtype):
    if obj is None: return None
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.dtype != target_dtype:
            return obj.to(target_dtype)
        return obj
    if isinstance(obj, list): return [_convert_to_dtype(x, target_dtype) for x in obj]
    if isinstance(obj, tuple): return tuple(_convert_to_dtype(x, target_dtype) for x in obj)
    if isinstance(obj, dict): return {k: _convert_to_dtype(v, target_dtype) for k, v in obj.items()}
    return obj

def _get_model_dtype(module):
    for param in module.parameters():
        if param.is_floating_point(): return param.dtype
    return torch.float32

def _patched_rope_attention_forward(self, q, k, v, num_k_exclude_rope=0):
    with torch.cuda.amp.autocast(enabled=False):
        target_dtype = _get_model_dtype(self)
        q, k, v = [_convert_to_dtype(x, target_dtype) for x in [q, k, v]]
        return _original_rope_attention_forward(self, q, k, v, num_k_exclude_rope)

def _patched_memory_attention_layer_forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
    with torch.cuda.amp.autocast(enabled=False):
        target_dtype = _get_model_dtype(self)
        tgt, memory, pos, query_pos = [_convert_to_dtype(x, target_dtype) for x in [tgt, memory, pos, query_pos]]
        return _original_memory_attention_layer_forward(self, tgt, memory, pos, query_pos, num_k_exclude_rope)

def _patched_memory_attention_forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
    with torch.cuda.amp.autocast(enabled=False):
        target_dtype = _get_model_dtype(self)
        curr, memory, curr_pos, memory_pos = [_convert_to_dtype(x, target_dtype) for x in [curr, memory, curr_pos, memory_pos]]
        return _original_memory_attention_forward(self, curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens)

RoPEAttention.forward = _patched_rope_attention_forward
MemoryAttentionLayer.forward = _patched_memory_attention_layer_forward
MemoryAttention.forward = _patched_memory_attention_forward

print(" SAM2 dtype patch applied!")

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



# STEP 2: CONFIGURATION


if torch.cuda.is_available():
    DEVICE_STR = "cuda"
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Device: CUDA")
else:
    DEVICE_STR = "cpu"
    DEVICE = torch.device("cpu")
    print(f"Device: CPU")

print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

VIDEO_PATH      = os.path.join(BASE_DIR, "data", "IMG_2763.mp4")
FRAME_DIR       = os.path.join(BASE_DIR, "frames")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")

RAW_FILE        = os.path.join(RESULTS_DIR, "tracking_results.pkl")
ALIGNED_FILE    = os.path.join(RESULTS_DIR, "aligned_positions.json")
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

# Landmark IDs
LANDMARK_NAS = 0    # Nasion (front)
LANDMARK_LPA = 1    # Left preauricular (left ear)
LANDMARK_RPA = 2    # Right preauricular (right ear)
NUM_LANDMARKS = 3   # Only 3 clicks needed

LANDMARK_NAMES = {
    LANDMARK_NAS: "NAS (Nasion - front of head)",
    LANDMARK_LPA: "LPA (Left ear)",
    LANDMARK_RPA: "RPA (Right ear)",
}

CONFIG = {
    "frame_skip": 10,
    "flash_duration": 3.0,
    "display_height": 800,
    "duplicate_radius": 40,
    "yolo_conf": 0.25,
    "cap_mask_expansion": 1.10,
    "cap_min_area_frac": 0.03,
    "cap_max_area_frac": 0.70,
    "cap_confirm_alpha": 0.45,
    "cap_autodetect_use_yolo": True,
    # Alignment settings
    "min_landmarks_for_alignment": 3,  # Minimum landmarks needed per frame
    "alignment_outlier_std": 2.0,      # Remove positions > N std from mean
}



# STEP 3: MODEL INITIALIZATION


def initialize_models():
    print(f"\nLoading YOLO from {YOLO_WEIGHTS}...")
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"ERROR: YOLO weights not found"); sys.exit(1)
    yolo = YOLO(YOLO_WEIGHTS)
    print("Loading SAM2 Video Predictor...")
    sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE_STR)
    sam2_predictor = sam2_predictor.float()
    sam2_predictor.eval()
    print("Models initialized\n")
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
    """Draw heads-up display with landmark/electrode status"""
    msg1 = f"Frame: {idx}/{total-1}"
    
    # Landmark status
    if current_id < NUM_LANDMARKS:
        status_parts = []
        for lid in range(NUM_LANDMARKS):
            name = LANDMARK_NAMES[lid].split(" ")[0]
            status = "[Done]" if current_id > lid else ("[Click]" if current_id == lid else "[Wait]")
            status_parts.append(f"{name}: {status}")
        msg2 = " | ".join(status_parts)
        color = (0, 255, 255)
    else:
        msg2 = f"Landmarks: [All Saved] | Electrodes: {current_id - NUM_LANDMARKS}"
        color = (0, 255, 0)
    
    cv2.rectangle(img, (0, 0), (img.shape[1], 170), (0, 0, 0), -1)
    cv2.putText(img, msg1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, msg2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Current landmark hint
    if current_id < NUM_LANDMARKS:
        hint = f">>> Click: {LANDMARK_NAMES[current_id]}"
        cv2.putText(img, hint, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(img, "NAV: [A] Back (-15) | [S] Fwd (+15) | [Q] Quit", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "ACT: [D] YOLO Detect | [R] Refresh Map | [Click] Add | [Space] Done", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    cv2.putText(img, "TIP: If click doesn't register, try a different frame (press S or A)", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    return img


def draw_reference_window(img, scale, points, new_detections=None):
    """Draw reference window showing all detected points"""
    disp, _ = resize_for_display(img, CONFIG["display_height"])
    ref_img = disp.copy()
    
    cv2.rectangle(ref_img, (0, 0), (ref_img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(ref_img, "DETECTION MAP - Spot missing electrodes", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ref_img, "GREEN=Landmarks | BLUE=Existing | RED=New YOLO", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw existing points
    for i, (px, py) in enumerate(points):
        dx, dy = int(px * scale), int(py * scale)
        if i < NUM_LANDMARKS:
            # Landmarks - green squares
            cv2.rectangle(ref_img, (dx-6, dy-6), (dx+6, dy+6), (0, 255, 0), -1)
            names = ["N", "L", "R"]  # Short names
            cv2.putText(ref_img, names[i], (dx+10, dy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # Electrodes - blue dots
            cv2.circle(ref_img, (dx, dy), 4, (255, 100, 0), -1)
    
    # Draw new YOLO detections
    if new_detections:
        for (x1, y1, x2, y2, eid) in new_detections:
            cx = int(((x1 + x2) / 2) * scale)
            cy = int(((y1 + y2) / 2) * scale)
            cv2.circle(ref_img, (cx, cy), 4, (0, 0, 255), -1)
    
    # Footer
    num_electrodes = max(0, len(points) - NUM_LANDMARKS)
    footer = f"Total: {NUM_LANDMARKS} landmarks, {num_electrodes} electrodes"
    cv2.putText(ref_img, footer, (10, ref_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return ref_img


def interactive_crop(video_path, display_height):
    """Let user select crop region"""
    cap = cv2.VideoCapture(video_path)
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
    """Extract frames from video"""
    if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
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


def is_duplicate(pt, existing, radius):
    """Check if point is too close to existing electrodes"""
    # Only check electrodes (skip landmarks)
    return any(np.linalg.norm(np.array(pt) - np.array(p)) < radius for p in existing[NUM_LANDMARKS:])


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
        _, _, logits = sam2.add_new_points_or_box(state, frame_idx=fidx, obj_id=999,
                                                 points=[np.array(clicks[0], dtype=np.float32)], labels=[1])
    return (logits[0] > 0).cpu().numpy().squeeze().astype(bool)

def auto_cap_mask(img, yolo=None):
    print("[Auto-Mask] Generating...")
    model = build_sam2_float32()
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.88,
                                     stability_score_thresh=0.92, min_mask_region_area=500)
    masks = amg.generate(img)
    del amg, model; gc.collect()
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
    mode = "manual"
    while True:
        if mode == "auto": mask = auto_cap_mask(img, yolo)
        else: mask = manual_cap_mask(img, sam2, state, 0, CONFIG["display_height"])
        mask_exp = expand_mask(mask, CONFIG["cap_mask_expansion"])
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        preview = draw_mask_overlay(disp, mask_exp, CONFIG["cap_confirm_alpha"])
        win = "Confirm Cap Mask"
        cv2.namedWindow(win)
        while True:
            show = preview.copy()
            cv2.rectangle(show, (0, 0), (show.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(show, f"Mode={mode} | y=OK | r=redo | m=manual | a=auto | q=quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(show, "Tip: Press 'm' and click center of cap", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
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


# STEP 6: FRAME ALIGNMENT & COORDINATE AVERAGING


# INION ESTIMATION (2D)


def estimate_inion_2d(nas, lpa, rpa, ratio=1.0):
    """
    Estimate INION position from NAS, LPA, RPA in 2D.
    
    Geometry:
    - Origin = midpoint of LPA-RPA
    - INION lies on the line perpendicular to LPA-RPA, passing through origin
    - INION is on the opposite side of NAS from the origin
    - Distance from origin to INION ≈ distance from origin to NAS (adjustable by ratio)
    
    Args:
        nas: (x, y) nasion position
        lpa: (x, y) left preauricular position
        rpa: (x, y) right preauricular position
        ratio: INION distance relative to NAS distance (default 1.0)
               Typical values: 0.9-1.1 depending on head shape
    
    Returns:
        (x, y) estimated INION position
    """
    nas = np.array(nas)
    lpa = np.array(lpa)
    rpa = np.array(rpa)
    
    # Origin: midpoint between ears
    origin = (lpa + rpa) / 2
    
    # Vector from origin to NAS
    nas_vec = nas - origin
    nas_dist = np.linalg.norm(nas_vec)
    
    if nas_dist < 1e-6:
        # NAS is at origin (shouldn't happen), can't estimate
        return None
    
    # INION is in the opposite direction
    inion_vec = -nas_vec / nas_dist  # Unit vector pointing to INION
    inion_dist = nas_dist * ratio    # Distance to INION
    
    inion = origin + inion_vec * inion_dist
    
    return inion


def compute_head_transform(landmarks):
    """
    Compute transformation matrix from landmarks to standard head coordinates.
    INION is estimated from NAS, LPA, RPA.
    
    Standard coordinate system:
    - Origin: Midpoint between LPA and RPA
    - X-axis: LPA -> RPA (left to right)
    - Y-axis: INION -> NAS (back to front)
    
    Args:
        landmarks: dict with keys LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA
    
    Returns:
        Transform dict or None if insufficient landmarks
    """
    required = [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]
    if not all(lid in landmarks and landmarks[lid] is not None for lid in required):
        return None
    
    NAS = np.array(landmarks[LANDMARK_NAS])
    LPA = np.array(landmarks[LANDMARK_LPA])
    RPA = np.array(landmarks[LANDMARK_RPA])
    
    # Estimate INION
    INION = estimate_inion_2d(NAS, LPA, RPA, ratio=1.0)
    
    # Origin: midpoint between ears
    origin = (LPA + RPA) / 2
    
    # X-axis: left to right
    x_axis = RPA - LPA
    x_len = np.linalg.norm(x_axis)
    if x_len < 1e-6:
        return None
    x_axis = x_axis / x_len
    
    # Y-axis: INION to NAS (back to front)
    y_axis = NAS - INION
    y_len = np.linalg.norm(y_axis)
    if y_len < 1e-6:
        return None
    
    # Orthogonalize Y to X (ensure perpendicular)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build 2D rotation matrix
    R = np.array([x_axis, y_axis])  # 2x2
    
    # Scale: normalize ear-to-ear distance to 1.0
    scale = 1.0 / x_len
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "ear_distance": x_len,
        "nas_to_inion": y_len,
        "estimated_inion": INION
    }

def transform_point(point, transform):
    """Apply transformation to a 2D point."""
    if transform is None or point is None:
        return None
    
    p = np.array(point)
    origin = transform["origin"]
    R = transform["rotation"]
    scale = transform["scale"]
    
    p_centered = p - origin
    p_rotated = R @ p_centered
    p_scaled = p_rotated * scale
    
    return p_scaled


def align_and_average_positions(tracking_data, min_landmarks=3):
    """
    Align all frames to a common head coordinate system and average positions.
    
    Args:
        tracking_data: {frame_idx: {electrode_id: (x, y), ...}, ...}
        min_landmarks: Minimum landmarks needed per frame for alignment
    
    Returns:
        - aligned_positions: {electrode_id: (x, y)} averaged head-centered coordinates
        - stats: alignment statistics
    """
    print(" Frame Alignment & Averaging ")
    
    # Collect all positions per electrode in head coordinates
    electrode_positions = {}  # {electrode_id: [(x, y), ...]}
    
    frames_used = 0
    frames_skipped = 0
    
    for fidx, frame_data in tracking_data.items():
        # Extract landmarks for this frame
        landmarks = {}
        for lid in range(NUM_LANDMARKS):
            if lid in frame_data and frame_data[lid] is not None:
                landmarks[lid] = frame_data[lid]
        
        # Compute transformation
        transform = compute_head_transform(landmarks)
        
        if transform is None:
            frames_skipped += 1
            continue
        
        frames_used += 1
        
        # Transform all electrodes in this frame
        for eid, pos in frame_data.items():
            if pos is None:
                continue
            
            # Transform to head coordinates
            pos_transformed = transform_point(pos, transform)
            
            if pos_transformed is not None:
                if eid not in electrode_positions:
                    electrode_positions[eid] = []
                electrode_positions[eid].append(pos_transformed)
    
    print(f"  Frames used for alignment: {frames_used}")
    print(f"  Frames skipped (insufficient landmarks): {frames_skipped}")
    
    # Average positions with outlier removal
    final_positions = {}
    
    for eid, positions in electrode_positions.items():
        if len(positions) == 0:
            continue
        
        pts = np.array(positions)
        
        # Outlier removal
        if len(pts) > 5:
            mean = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            threshold = np.mean(dists) + CONFIG["alignment_outlier_std"] * np.std(dists)
            mask = dists < threshold
            if np.sum(mask) >= 3:
                pts = pts[mask]
        
        # Final average
        final_pos = np.mean(pts, axis=0)
        final_positions[eid] = final_pos
        
        # Label
        if eid < NUM_LANDMARKS:
            label = ["NAS", "LPA", "RPA", "INION"][eid]
        else:
            label = f"E{eid - NUM_LANDMARKS}"
        
        print(f"  {label}: {len(positions)} observations -> ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    
    stats = {
        "frames_used": frames_used,
        "frames_skipped": frames_skipped,
        "total_electrodes": len([k for k in final_positions if k >= NUM_LANDMARKS]),
        "total_landmarks": len([k for k in final_positions if k < NUM_LANDMARKS])
    }
    
    return final_positions, stats


def save_aligned_positions(positions, stats):
    """Save aligned positions to JSON."""
    
    output = {
        "coordinate_system": "head_aligned_2d",
        "description": "2D positions in head-centered coordinates (origin=ear midpoint, X=left-right, Y=back-front)",
        "scale": "normalized (ear-to-ear = 1.0)",
        "stats": stats,
        "landmarks": {},
        "electrodes": {}
    }
    
    for eid, pos in positions.items():
        pos_list = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
        
        if eid == LANDMARK_NAS:
            output["landmarks"]["NAS"] = pos_list
        elif eid == LANDMARK_LPA:
            output["landmarks"]["LPA"] = pos_list
        elif eid == LANDMARK_RPA:
            output["landmarks"]["RPA"] = pos_list
        else:
            output["electrodes"][f"E{eid - NUM_LANDMARKS}"] = pos_list
    
    with open(ALIGNED_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved aligned positions: {ALIGNED_FILE}")


def order_electrodes(positions):
    """Order electrodes by position (front to back, left to right)."""
    electrode_list = []
    
    for eid, pos in positions.items():
        if eid >= NUM_LANDMARKS:
            electrode_list.append({
                "id": eid,
                "x": pos[0] if isinstance(pos, np.ndarray) else pos[0],
                "y": pos[1] if isinstance(pos, np.ndarray) else pos[1]
            })
    
    # Sort: front to back (higher Y first), then left to right (lower X first)
    sorted_electrodes = sorted(electrode_list, key=lambda e: (-e["y"], e["x"]))
    
    return [e["id"] for e in sorted_electrodes]



# STEP 7: MAIN PIPELINE


def main():
    print("\n" + "=" * 70)
    print("EEG ELECTRODE REGISTRATION PIPELINE (ENHANCED)")
    print("with Frame Alignment & Coordinate Averaging")
    print("=" * 70 + "\n")

    # 1. Initialize models
    yolo, sam2 = initialize_models()

    # 2. Extract frames
    frames, (off_x, off_y) = extract_frames()
    if not frames: return

    # 3. Initialize SAM2 state
    print("\nInitializing SAM2 state...")
    with torch.inference_mode():
        state = sam2.init_state(video_path=FRAME_DIR)

    # 4. Create cap mask
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
    
    points = []
    current_id = 0
    idx_box = [0]
    flashes = []

    def click(e, x, y, f, p):
        nonlocal current_id
        if e != cv2.EVENT_LBUTTONDOWN: return
        
        cidx = idx_box[0]
        rx, ry = int(x/SCALE), int(y/SCALE)

        cap = cap_cache.get(cidx)
        if current_id >= NUM_LANDMARKS:
            if cap is None or not (0 <= ry < cap.shape[0] and 0 <= rx < cap.shape[1] and cap[ry, rx]):
                print(">>> Outside cap mask - rejected")
                return
            if is_duplicate((rx, ry), points, CONFIG["duplicate_radius"]):
                print(">>> Too close to existing electrode - rejected")
                return

        with torch.inference_mode():
            sam2.add_new_points_or_box(state, frame_idx=cidx, obj_id=current_id,
                                       points=[np.array((rx, ry), dtype=np.float32)], labels=[1])
        
        points.append((rx, ry))
        
        color = (0, 255, 0) if current_id < NUM_LANDMARKS else (0, 0, 255)
        flashes.append((rx, ry, color, time.time()))
        
        if current_id < NUM_LANDMARKS:
            print(f"Added landmark {LANDMARK_NAMES[current_id]} at ({rx}, {ry})")
        else:
            print(f"Added electrode E{current_id - NUM_LANDMARKS} at ({rx}, {ry})")
        current_id += 1

    cv2.namedWindow("Pipeline")
    cv2.setMouseCallback("Pipeline", click)

    print("\n--- Interactive Phase ---")
    print(f"1. Click {NUM_LANDMARKS} landmarks: NAS, LPA, RPA")
    print("2. Press 'D' to auto-detect electrodes with YOLO")
    print("3. Click manually to add missed electrodes")
    print("4. Press SPACE when done")
    
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

        if k == ord('s'):
            idx_box[0] = min(cidx + 15, len(frames) - 1)
        elif k == ord('a'):
            idx_box[0] = max(cidx - 15, 0)
        elif k == ord('q'):
            sys.exit(0)
        elif k == ord('d'):
            if current_id < NUM_LANDMARKS:
                print(f">>> Click all {NUM_LANDMARKS} landmarks first!")
                continue
            
            print(f"\nRunning YOLO detection on frame {cidx}...")
            res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
            
            new_detections = []
            found = 0
            
            if res and res[0].boxes is not None:
                cap = cap_cache.get(cidx)
                for box in res[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    if cap is not None:
                        if not (0 <= int(cy) < cap.shape[0] and 0 <= int(cx) < cap.shape[1] and cap[int(cy), int(cx)]):
                            continue
                    
                    if is_duplicate((cx, cy), points, CONFIG["duplicate_radius"]):
                        continue
                    
                    with torch.inference_mode():
                        sam2.add_new_points_or_box(state, frame_idx=cidx, obj_id=current_id, box=box)
                    
                    points.append((cx, cy))
                    new_detections.append((x1, y1, x2, y2, current_id))
                    flashes.append((cx, cy, (0, 0, 255), time.time()))
                    current_id += 1
                    found += 1
            
            print(f"✓ Added {found} new electrodes")
            ref_img = draw_reference_window(img, SCALE, points, new_detections)
            cv2.imshow("Reference Map", ref_img)
            
        elif k == ord('r'):
            if current_id >= NUM_LANDMARKS:
                ref_img = draw_reference_window(img, SCALE, points, None)
                cv2.imshow("Reference Map", ref_img)
                
        elif k == 32:  # Space
            if current_id < NUM_LANDMARKS:
                print(f">>> Must click all {NUM_LANDMARKS} landmarks first")
                continue
            try: cv2.destroyWindow("Reference Map")
            except: pass
            break

    cv2.destroyAllWindows()

    # 7. SAM2 Tracking
    print("\n--- SAM2 Tracking ---")
    print(f"Tracking {current_id} objects across {len(frames)} frames...")
    
    tracking = {}
    with torch.inference_mode():
        for fidx, ids, logits in tqdm(sam2.propagate_in_video(state), total=len(frames)):
            fd = {}
            for i, oid in enumerate(ids):
                if oid == 999: continue
                m = (logits[i] > 0).cpu().numpy().squeeze()
                ys, xs = np.where(m)
                if len(xs) > 0:
                    fd[int(oid)] = (float(np.mean(xs)), float(np.mean(ys))) 
            tracking[int(fidx)] = fd

    # 8. Save raw tracking
    print("\nSaving raw tracking results...")
    with open(RAW_FILE, "wb") as f:
        pickle.dump(tracking, f)
    print(f" Saved: {RAW_FILE}")

    # 9. Frame alignment and averaging
    aligned_positions, stats = align_and_average_positions(tracking)
    
    # 10. Save aligned positions
    save_aligned_positions(aligned_positions, stats)

    # 11. Order electrodes
    ordered = order_electrodes(aligned_positions)
    if ordered:
        with open(ORDER_FILE, "w") as f:
            json.dump(ordered, f)
        print(f" Saved electrode order: {ORDER_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved:")
    print(f"  - Raw tracking:      {RAW_FILE}")
    print(f"  - Aligned positions: {ALIGNED_FILE}")
    print(f"  - Electrode order:   {ORDER_FILE}")
    print(f"\nStatistics:")
    print(f"  - Frames used:   {stats['frames_used']}")
    print(f"  - Landmarks:     {stats['total_landmarks']}")
    print(f"  - Electrodes:    {stats['total_electrodes']}")


if __name__ == "__main__":
    main()