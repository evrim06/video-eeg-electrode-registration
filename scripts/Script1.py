"""
SCRIPT 1: ANNOTATION & TRACKING

KEY INSIGHT: Head is STATIONARY, only camera moves.
             -> We can PROJECT 3D points to 2D instead of tracking in 2D.

FEATURES:
    - YOLO confidence filtering (80% threshold) - Only accept high-quality detections
    - Multi-view triangulation - Better 3D accuracy from diverse angles
    - Subprocess isolation for VGGT - Prevents memory crashes with 28+ frames
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
import glob
import subprocess
from tqdm import tqdm
from ultralytics import YOLO
import colorsys


print("SCRIPT 1: ANNOTATION & TRACKING")


# CONFIGURATION


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Device: GPU (CUDA)")
else:
    print(f"Device: CPU (VGGT will take 30-60+ minutes - be patient)")

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

VIDEO_DIR = os.path.join(BASE_DIR, "data", "Video_Recordings")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VGGT_REPO_PATH = os.path.join(SCRIPT_DIR, "vggt")

YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")


def setup_video_output_dir(video_path):
    """
    Create video-specific output directory and return all output paths.
    
    Args:
        video_path: Full path to video file
        
    Returns:
        dict with all output paths for this video
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video_results_dir = os.path.join(RESULTS_BASE_DIR, video_name)
    vggt_output_dir = os.path.join(video_results_dir, "vggt_output")
    temp_vggt_dir = os.path.join(BASE_DIR, "data", f"vggt_ready_frames_{video_name}")
    
    os.makedirs(video_results_dir, exist_ok=True)
    os.makedirs(vggt_output_dir, exist_ok=True)
    
    paths = {
        "video_name": video_name,
        "results_dir": video_results_dir,
        "vggt_output_dir": vggt_output_dir,
        "temp_vggt_dir": temp_vggt_dir,
        "tracking_file": os.path.join(video_results_dir, "tracking_results.pkl"),
        "crop_file": os.path.join(video_results_dir, "crop_info.json"),
        "points_3d_file": os.path.join(video_results_dir, "points_3d_intermediate.pkl"),
    }
    
    print(f"OUTPUT DIRECTORY SETUP")
    print(f"  Video: {video_name}")
    print(f"  Results directory: {video_results_dir}")
    print(f"  All outputs will be saved in this folder")
    
    return paths

print(f"Base directory: {BASE_DIR}")
print(f"Video directory: {VIDEO_DIR}")
print(f"Results base directory: {RESULTS_BASE_DIR}")

# Object IDs
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3
ELECTRODE_START_ID = 100

LANDMARK_NAMES = {0: "NAS (Nasion)", 1: "LPA (Left ear)", 2: "RPA (Right ear)"}
LANDMARK_SHORT = {0: "NAS", 1: "LPA", 2: "RPA"}
LANDMARK_COLORS = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}

VGGT_SIZE = 518
MAX_VGGT_FRAMES = 28  # Same for CPU and GPU - no timeout, will wait for completion

CONFIG = {
    "frame_skip": 5,
    "display_height": 700,
    "yolo_conf": 0.25,              # Minimum confidence for YOLO to detect
    "yolo_conf_accept": 0.80,       # Minimum confidence to accept detection (quality filter)
    "mask_alpha": 0.5,
    "playback_fps": 5,
    "projection_search_radius": 30,
    "min_triangulation_angle": 5.0,
    "visibility_depth_threshold": 0.1,
    "click_match_radius": 20,       # Radius for matching clicks to existing 2D annotations
    "projection_match_radius": 40,  # Radius for 2D projection matching (fallback only)
    "match_3d_head_ratio": 0.12,    # 3D match threshold as ratio of ear-to-ear distance (~12%)
}


# HELPER FUNCTIONS


def get_color(obj_id):
    if obj_id in LANDMARK_COLORS:
        return LANDMARK_COLORS[obj_id]
    i = obj_id - ELECTRODE_START_ID if obj_id >= ELECTRODE_START_ID else obj_id
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))


def resize_for_display(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h)), scale


def script1_to_vggt_coords(u, v, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


def vggt_to_script1_coords(u_vggt, v_vggt, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return (u_vggt - pad_w) / scale, (v_vggt - pad_h) / scale


def unproject_pixel(u, v, depth_map, intrinsic, extrinsic):
    H, W = depth_map.shape
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
    du, dv = u - u0, v - v0
    z = (depth_map[v0, u0] * (1 - du) * (1 - dv) +
         depth_map[v0, u1] * du * (1 - dv) +
         depth_map[v1, u0] * (1 - du) * dv +
         depth_map[v1, u1] * du * dv)
    if z <= 0 or not np.isfinite(z):
        return None
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    P_cam = np.array([x_cam, y_cam, z, 1.0])
    P_world = (np.linalg.inv(extrinsic) @ P_cam)[:3]
    return P_world


def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    P_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    P_cam = extrinsic @ P_world
    x_cam, y_cam, z_cam = P_cam[0], P_cam[1], P_cam[2]
    if z_cam <= 0:
        return None, False
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    return (u, v), True


# VIDEO SELECTION AND FRAME EXTRACTION


def select_video():
    extensions = ['*.mp4', '*.MP4', '*.MOV', '*.mov']
    all_videos = []
    for ext in extensions:
        all_videos.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    seen = set()
    videos = []
    for v in sorted(all_videos):
        if v.lower() not in seen:
            seen.add(v.lower())
            videos.append(v)
    if not videos:
        print(f"ERROR: No videos in {VIDEO_DIR}")
        sys.exit(1)
    print("\n--- Select Video ---")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")
    choice = input(f"\nEnter (1-{len(videos)}): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            return videos[idx]
    except:
        pass
    return videos[0]


def extract_frames(video_path, output_paths):
    """Extract frames from video and save crop info."""
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, roi, drawing, start = 0, None, False, None
    
    def mouse(e, x, y, f, p):
        nonlocal roi, drawing, start
        if e == cv2.EVENT_LBUTTONDOWN:
            drawing, start, roi = True, (x,y), None
        elif e == cv2.EVENT_MOUSEMOVE and drawing:
            roi = (min(start[0],x), min(start[1],y), abs(x-start[0]), abs(y-start[1]))
        elif e == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    cv2.namedWindow("Crop")
    cv2.setMouseCallback("Crop", mouse)
    print("\n--- Draw box around HEAD ---")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        disp, scale = resize_for_display(frame, CONFIG["display_height"])
        show = disp.copy()
        cv2.putText(show, f"Frame {idx}/{total-1} | A/S:Nav | SPACE:Confirm", (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if roi:
            cv2.rectangle(show, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,255), 2)
        cv2.imshow("Crop", show)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('a'): idx = max(0, idx-15)
        elif k == ord('s'): idx = min(total-1, idx+15)
        elif k in (13,32) and roi: break
        elif k == ord('q'): sys.exit(0)
    
    cv2.destroyWindow("Crop")
    cap.release()
    
    sx, sy, sw, sh = int(roi[0]/scale), int(roi[1]/scale), int(roi[2]/scale), int(roi[3]/scale)
    cap = cv2.VideoCapture(video_path)
    skip = CONFIG["frame_skip"]
    frames, fidx, count = [], 0, 0
    print(f"\nExtracting frames (skip={skip})...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % skip == 0:
            crop = frame[sy:sy+sh, sx:sx+sw]
            cv2.imwrite(os.path.join(FRAME_DIR, f"{fidx:05d}.jpg"), crop)
            frames.append(f"{fidx:05d}.jpg")
            fidx += 1
        count += 1
    cap.release()
    
    crop_info = {"x": sx, "y": sy, "w": sw, "h": sh, "skip": skip}
    crop_info_file = output_paths["crop_file"]
    with open(crop_info_file, "w") as f:
        json.dump(crop_info, f)
    
    print(f"Extracted {len(frames)} frames")
    return frames, crop_info


# VGGT RECONSTRUCTION


def run_vggt_full(frames, crop_info, output_paths):
    """
    Run VGGT on frames to get camera poses.
    Uses subprocess isolation to prevent memory accumulation.
    """
    print("\n=== RUNNING VGGT FOR CAMERA POSES ===")
    
    vggt_output_dir = output_paths["vggt_output_dir"]
    temp_vggt_dir = output_paths["temp_vggt_dir"]
    
    os.makedirs(vggt_output_dir, exist_ok=True)
    
    # Select frames for VGGT
    all_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    n = min(len(all_paths), MAX_VGGT_FRAMES)
    indices = np.linspace(0, len(all_paths)-1, n, dtype=int)
    selected = [all_paths[i] for i in indices]
    s1_indices = [int(os.path.basename(p).split('.')[0]) for p in selected]
    print(f"  Processing {n} frames for VGGT...")
    
    # Prepare frames for VGGT (resize and pad to 518x518)
    if os.path.exists(temp_vggt_dir): 
        shutil.rmtree(temp_vggt_dir)
    os.makedirs(temp_vggt_dir)
    
    print("  Preparing frames...")
    for i, path in enumerate(selected):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        scale = VGGT_SIZE / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        pad_w = (VGGT_SIZE - img.shape[1]) // 2
        pad_h = (VGGT_SIZE - img.shape[0]) // 2
        img = cv2.copyMakeBorder(img, pad_h, VGGT_SIZE-img.shape[0]-pad_h,
                                 pad_w, VGGT_SIZE-img.shape[1]-pad_w,
                                 cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.imwrite(os.path.join(temp_vggt_dir, f"{i:05d}.jpg"), img)
    
    # Save frame mapping for subprocess
    np.save(os.path.join(temp_vggt_dir, "s1_indices.npy"), np.array(s1_indices))
    
    # Run VGGT in subprocess (isolated memory space)
    print("  Launching VGGT subprocess...")
    result = _run_vggt_subprocess(temp_vggt_dir, vggt_output_dir, CHECKPOINT_DIR, VGGT_REPO_PATH)
    
    if not result:
        print("\n  ERROR: VGGT subprocess failed!")
        print("  Check the error messages above.")
        print("  Common issues: not enough RAM, missing dependencies")
        sys.exit(1)
    
    # Load results from subprocess
    print("  Loading VGGT results...")
    recon = np.load(os.path.join(vggt_output_dir, "reconstruction.npz"))
    
    depth = recon["depth"]
    E = recon["extrinsics"]
    K = recon["intrinsics"]
    
    if E.shape[-2:] == (3,4):
        E4 = np.zeros((len(s1_indices), 4, 4))
        for i in range(len(s1_indices)):
            E4[i] = np.eye(4)
            E4[i,:3,:] = E[i]
        E = E4
    
    print(f"[OK] VGGT complete - {len(s1_indices)} frames processed")
    
    vggt_data = {
        "depths": depth,
        "extrinsics": E,
        "intrinsics": K,
        "s1_to_vggt": {int(s1): int(vggt) for vggt, s1 in enumerate(s1_indices)},
        "vggt_to_s1": {int(vggt): int(s1) for vggt, s1 in enumerate(s1_indices)},
        "s1_indices": s1_indices,
    }
    return vggt_data


def _run_vggt_subprocess(temp_vggt_dir, vggt_output_dir, checkpoint_dir, vggt_repo_path):
    """
    Run VGGT inference in a separate subprocess.
    This ensures ALL memory (GPU + CPU) is released when done.
    """
    
    # Create a temporary Python script for the subprocess
    # Use repr() for paths to handle Windows backslashes properly
    subprocess_script = f'''
import sys
import os
import gc
import numpy as np
import torch
import glob

# Configuration - paths use repr() to handle Windows backslashes
TEMP_DIR = {repr(temp_vggt_dir)}
OUTPUT_DIR = {repr(vggt_output_dir)}
CHECKPOINT_DIR = {repr(checkpoint_dir)}
VGGT_REPO = {repr(vggt_repo_path)}

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable TF32 for numerical precision (only affects CUDA)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

print(f"  VGGT Subprocess started (Device: {{DEVICE}})")
if DEVICE == "cpu":
    print("  NOTE: Running on CPU - this will take several minutes")

try:
    # Import VGGT
    sys.path.insert(0, VGGT_REPO)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    # Clear memory before loading model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        print(f"  GPU Memory: {{free_mem:.1f}}GB free / {{total_mem:.1f}}GB total")
    
    # Load checkpoint
    checkpoint = os.path.join(CHECKPOINT_DIR, "vggt_model.pt")
    model = VGGT()
    
    if not os.path.exists(checkpoint):
        print("  Downloading VGGT weights (this may take a while)...")
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location="cpu", progress=True)
        torch.save(state, checkpoint)
        del state
        gc.collect()
    
    print("  Loading VGGT model...")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()
    
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Load images
    image_paths = sorted(glob.glob(os.path.join(TEMP_DIR, "*.jpg")))
    print(f"  Loading {{len(image_paths)}} images...")
    images = load_and_preprocess_images(image_paths).to(DEVICE)
    print(f"  Image tensor shape: {{images.shape}}")
    
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Run inference
    print("  Running VGGT inference...")
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds = model(images)
        else:
            # CPU inference - no autocast
            preds = model(images)
    
    print("  Processing outputs...")
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    depth = preds["depth"].float().cpu().numpy().squeeze()
    E = E.float().cpu().numpy().squeeze()
    K = K.float().cpu().numpy().squeeze()
    
    # Load frame mapping
    s1_indices = np.load(os.path.join(TEMP_DIR, "s1_indices.npy"))
    
    # Save results
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "reconstruction.npz"),
        depth=depth, extrinsics=E, intrinsics=K,
        frame_mapping_keys=np.arange(len(s1_indices)),
        frame_mapping_values=s1_indices
    )
    
    print("  VGGT subprocess completed successfully")
    sys.exit(0)
    
except Exception as e:
    print(f"  VGGT subprocess error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    # Write subprocess script to temp file
    script_path = os.path.join(temp_vggt_dir, "vggt_subprocess.py")
    with open(script_path, "w") as f:
        f.write(subprocess_script)
    
    # Run subprocess - no timeout, let it complete naturally
    # Subprocess isolation still helps with memory cleanup after completion
    print("  (No timeout - will run until complete)")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False  # Show output in real-time
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n  Interrupted by user (Ctrl+C)")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to run subprocess: {e}")
        return False


# MODEL LOADING


def load_yolo():
    """Load YOLO model for electrode detection."""
    print(f"\nLoading YOLO model...")
    yolo = None
    if os.path.exists(YOLO_WEIGHTS):
        yolo = YOLO(YOLO_WEIGHTS)
        print("  YOLO loaded successfully")
    else:
        print("  WARNING: YOLO not found, will use manual detection only")
    return yolo


# MULTI-VIEW ANNOTATOR


class MultiViewAnnotator:
    def __init__(self, frames, vggt_data, crop_info):
        self.frames = frames
        self.vggt_data = vggt_data
        self.crop_info = crop_info
        self.annotations_2d = {}
        self.positions_3d = {}
        img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
        _, self.scale = resize_for_display(img0, CONFIG["display_height"])
        self.img_h, self.img_w = img0.shape[:2]
        self.current_obj_id = None
        self.next_electrode_id = ELECTRODE_START_ID
    
    def get_vggt_frame_idx(self, s1_idx):
        return self.vggt_data["s1_to_vggt"].get(s1_idx, None)
    
    def unproject_click(self, u, v, s1_idx):
        vggt_idx = self.get_vggt_frame_idx(s1_idx)
        if vggt_idx is None:
            return None
        u_vggt, v_vggt = script1_to_vggt_coords(u, v, self.crop_info["w"], self.crop_info["h"])
        return unproject_pixel(
            u_vggt, v_vggt,
            self.vggt_data["depths"][vggt_idx],
            self.vggt_data["intrinsics"][vggt_idx],
            self.vggt_data["extrinsics"][vggt_idx]
        )
    
    def triangulate_point(self, obj_id):
        if obj_id not in self.annotations_2d:
            return None
        observations = self.annotations_2d[obj_id]
        if len(observations) < 1:
            return None
        points_3d = []
        weights = []
        for s1_idx, (u, v) in observations.items():
            vggt_idx = self.get_vggt_frame_idx(s1_idx)
            if vggt_idx is None:
                continue
            p3d = self.unproject_click(u, v, s1_idx)
            if p3d is None:
                continue
            weight = 1.0
            u_vggt, v_vggt = script1_to_vggt_coords(u, v, self.crop_info["w"], self.crop_info["h"])
            depth_map = self.vggt_data["depths"][vggt_idx]
            H, W = depth_map.shape
            if 0 <= int(v_vggt) < H and 0 <= int(u_vggt) < W:
                depth = depth_map[int(v_vggt), int(u_vggt)]
                if depth > 0:
                    weight *= 1.0 / (1.0 + depth)
            img_center_u, img_center_v = W / 2, H / 2
            dist_from_center = np.sqrt((u_vggt - img_center_u)**2 + (v_vggt - img_center_v)**2)
            max_dist = np.sqrt(img_center_u**2 + img_center_v**2)
            center_weight = 1.0 - 0.5 * (dist_from_center / max_dist)
            weight *= center_weight
            points_3d.append(p3d)
            weights.append(weight)
        if not points_3d:
            return None
        points_3d = np.array(points_3d)
        weights = np.array(weights)
        if len(points_3d) >= 3:
            median = np.median(points_3d, axis=0)
            dists = np.linalg.norm(points_3d - median, axis=1)
            threshold = np.mean(dists) + 2 * np.std(dists) if np.std(dists) > 0 else np.inf
            mask = dists < threshold
            if np.sum(mask) >= 1:
                valid_points = points_3d[mask]
                valid_weights = weights[mask]
                valid_weights = valid_weights / valid_weights.sum()
                return np.average(valid_points, axis=0, weights=valid_weights)
        weights = weights / weights.sum()
        return np.average(points_3d, axis=0, weights=weights)
    
    def project_to_frame(self, obj_id, s1_idx):
        if obj_id not in self.positions_3d:
            return None, False
        vggt_idx = self.get_vggt_frame_idx(s1_idx)
        if vggt_idx is None:
            return None, False
        point_3d = self.positions_3d[obj_id]
        proj, in_front = project_3d_to_2d(
            point_3d,
            self.vggt_data["intrinsics"][vggt_idx],
            self.vggt_data["extrinsics"][vggt_idx]
        )
        if not in_front:
            return None, False
        u_vggt, v_vggt = proj
        u_s1, v_s1 = vggt_to_script1_coords(u_vggt, v_vggt, self.crop_info["w"], self.crop_info["h"])
        if not (0 <= u_s1 < self.img_w and 0 <= v_s1 < self.img_h):
            return None, False
        return (u_s1, v_s1), True
    
    def update_3d_positions(self):
        for obj_id in self.annotations_2d:
            p3d = self.triangulate_point(obj_id)
            if p3d is not None:
                self.positions_3d[obj_id] = p3d
    
    def get_head_size_3d(self):
        """
        Get the head size in VGGT 3D units based on landmark positions.
        Returns the ear-to-ear distance (LPA to RPA).
        """
        if LANDMARK_LPA in self.positions_3d and LANDMARK_RPA in self.positions_3d:
            return np.linalg.norm(
                self.positions_3d[LANDMARK_LPA] - self.positions_3d[LANDMARK_RPA]
            )
        return None
    
    def find_closest_electrode_3d(self, u, v, s1_idx):
        """
        Find the closest existing electrode by comparing in 3D space.
        
        This is the PRIMARY matching method - more robust than 2D projection
        because it handles cases where the same electrode clicked from different
        viewpoints produces slightly different 3D estimates due to depth errors.
        
        The threshold is computed as a ratio of the head size (ear-to-ear distance),
        so it adapts to the scale of the reconstruction.
        
        Args:
            u, v: Click position in frame coordinates
            s1_idx: Frame index
        
        Returns:
            (obj_id, distance_3d) if found within threshold, else (None, inf)
        """
        # Unproject the click to 3D
        click_3d = self.unproject_click(u, v, s1_idx)
        if click_3d is None:
            return None, float('inf')
        
        # Compute threshold based on head size
        head_size = self.get_head_size_3d()
        if head_size is not None:
            max_dist_3d = head_size * CONFIG["match_3d_head_ratio"]
        else:
            # Fallback if landmarks not yet annotated (shouldn't happen normally)
            max_dist_3d = 0.1
        
        closest_id = None
        closest_dist = float('inf')
        
        for obj_id, pos_3d in self.positions_3d.items():
            if obj_id < ELECTRODE_START_ID:
                continue
            
            # Skip if this electrode already has an annotation in this frame
            if obj_id in self.annotations_2d and s1_idx in self.annotations_2d[obj_id]:
                continue
            
            # Compare in 3D space directly
            dist_3d = np.linalg.norm(click_3d - pos_3d)
            
            if dist_3d < closest_dist:
                closest_dist = dist_3d
                closest_id = obj_id
        
        if closest_dist <= max_dist_3d:
            return closest_id, closest_dist
        return None, float('inf')
    
    def find_closest_projected_electrode(self, u, v, s1_idx, max_radius=None):
        """
        Find the closest existing electrode whose 3D position projects near (u, v).
        
        Returns:
            (obj_id, distance) if found within max_radius, else (None, inf)
        """
        if max_radius is None:
            max_radius = CONFIG["projection_match_radius"]
        
        closest_id = None
        closest_dist = float('inf')
        
        for obj_id in self.positions_3d:
            if obj_id < ELECTRODE_START_ID:
                continue
            
            # Skip if this electrode already has an annotation in this frame
            if obj_id in self.annotations_2d and s1_idx in self.annotations_2d[obj_id]:
                continue
            
            proj, visible = self.project_to_frame(obj_id, s1_idx)
            if visible and proj:
                dist = np.sqrt((u - proj[0])**2 + (v - proj[1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = obj_id
        
        if closest_dist <= max_radius:
            return closest_id, closest_dist
        return None, float('inf')
    
    def annotate_landmarks(self):
        print("\n=== LANDMARK ANNOTATION ===")
        print("Navigate to frames where each landmark is clearly visible")
        print("Click to mark, annotations from multiple views improve accuracy")
        for landmark_id, name in [(LANDMARK_NAS, "NAS (Nasion)"), 
                                   (LANDMARK_LPA, "LPA (Left ear pin)"),
                                   (LANDMARK_RPA, "RPA (Right ear pin)")]:
            self._annotate_single_object(landmark_id, name, is_landmark=True)
        self.update_3d_positions()
        print("\n[OK] Landmarks annotated:")
        for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            n_obs = len(self.annotations_2d.get(lid, {}))
            has_3d = lid in self.positions_3d
            print(f"  {LANDMARK_SHORT[lid]}: {n_obs} observations, 3D: {'Yes' if has_3d else 'No'}")
    
    def annotate_electrodes(self, yolo=None):
        print("\n=== ELECTRODE ANNOTATION ===")
        print("Navigate through frames and click on electrodes")
        print("SPACE: Auto-detect with YOLO | Click: Add electrode | Right-click: Remove")
        print("Y: Done with this view | N: Next electrode | Q: Finish all")
        vggt_frames = sorted(self.vggt_data["s1_to_vggt"].keys())
        idx = 0
        cv2.namedWindow("Electrodes")
        click_pos = None
        
        def mouse(e, x, y, f, p):
            nonlocal click_pos
            if e == cv2.EVENT_LBUTTONDOWN:
                click_pos = (int(x / self.scale), int(y / self.scale))
        
        cv2.setMouseCallback("Electrodes", mouse)
        
        while True:
            s1_idx = vggt_frames[idx]
            frame_path = os.path.join(FRAME_DIR, self.frames[s1_idx])
            img = cv2.imread(frame_path)
            disp, _ = resize_for_display(img, CONFIG["display_height"])
            
            for obj_id in self.annotations_2d:
                if obj_id < ELECTRODE_START_ID:
                    continue
                color = get_color(obj_id)
                if s1_idx in self.annotations_2d[obj_id]:
                    u, v = self.annotations_2d[obj_id][s1_idx]
                    dx, dy = int(u * self.scale), int(v * self.scale)
                    cv2.circle(disp, (dx, dy), 8, color, -1)
                    cv2.circle(disp, (dx, dy), 10, (255, 255, 255), 2)
                    label = f"E{obj_id - ELECTRODE_START_ID}"
                    cv2.putText(disp, label, (dx + 12, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                elif obj_id in self.positions_3d:
                    proj, visible = self.project_to_frame(obj_id, s1_idx)
                    if visible and proj:
                        dx, dy = int(proj[0] * self.scale), int(proj[1] * self.scale)
                        cv2.circle(disp, (dx, dy), 6, color, 1)
                        label = f"E{obj_id - ELECTRODE_START_ID}*"
                        cv2.putText(disp, label, (dx + 12, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
                if lid in self.positions_3d:
                    proj, visible = self.project_to_frame(lid, s1_idx)
                    if visible and proj:
                        dx, dy = int(proj[0] * self.scale), int(proj[1] * self.scale)
                        cv2.circle(disp, (dx, dy), 5, LANDMARK_COLORS[lid], -1)
                        cv2.putText(disp, LANDMARK_SHORT[lid], (dx + 8, dy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, LANDMARK_COLORS[lid], 1)
            
            n_electrodes = len([k for k in self.annotations_2d if k >= ELECTRODE_START_ID])
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(disp, f"Frame {s1_idx} ({idx+1}/{len(vggt_frames)}) | Electrodes: {n_electrodes}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(disp, "A/D:Nav | Click:Add | SPACE:AutoDetect | Q:Done", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("Electrodes", disp)
            
            k = cv2.waitKey(1) & 0xFF
            
            if k in [ord('a'), 81]: 
                idx = max(0, idx - 1)
            elif k in [ord('d'), 83]: 
                idx = min(len(vggt_frames) - 1, idx + 1)
            elif k == ord('w'): 
                idx = max(0, idx - 5)
            elif k == ord('s'): 
                idx = min(len(vggt_frames) - 1, idx + 5)
            elif click_pos is not None:
                u, v = click_pos
                click_pos = None
                clicked_existing = False
                click_radius = CONFIG["click_match_radius"]
                
                # STEP 1: Check if clicking on existing 2D annotation in this frame (to remove)
                for obj_id in list(self.annotations_2d.keys()):
                    if obj_id < ELECTRODE_START_ID:
                        continue
                    if s1_idx in self.annotations_2d[obj_id]:
                        eu, ev = self.annotations_2d[obj_id][s1_idx]
                        if np.sqrt((u - eu)**2 + (v - ev)**2) < click_radius:
                            del self.annotations_2d[obj_id][s1_idx]
                            if len(self.annotations_2d[obj_id]) == 0:
                                del self.annotations_2d[obj_id]
                                if obj_id in self.positions_3d:
                                    del self.positions_3d[obj_id]
                            clicked_existing = True
                            print(f"  Removed E{obj_id - ELECTRODE_START_ID} from frame {s1_idx}")
                            break
                
                # STEP 2: Check 3D distance (most robust - handles different viewpoints)
                if not clicked_existing:
                    closest_id, closest_dist_3d = self.find_closest_electrode_3d(u, v, s1_idx)
                    if closest_id is not None:
                        self.annotations_2d[closest_id][s1_idx] = (u, v)
                        clicked_existing = True
                        print(f"  Added observation to E{closest_id - ELECTRODE_START_ID} (3D match, dist={closest_dist_3d:.3f})")
                
                # STEP 3: Fall back to 2D projection matching
                if not clicked_existing:
                    closest_id, closest_dist_2d = self.find_closest_projected_electrode(u, v, s1_idx)
                    if closest_id is not None:
                        self.annotations_2d[closest_id][s1_idx] = (u, v)
                        clicked_existing = True
                        print(f"  Added observation to E{closest_id - ELECTRODE_START_ID} (2D match, dist={closest_dist_2d:.1f}px)")
                
                # STEP 4: Create new electrode if no match
                if not clicked_existing:
                    new_id = self.next_electrode_id
                    self.next_electrode_id += 1
                    self.annotations_2d[new_id] = {s1_idx: (u, v)}
                    print(f"  Added E{new_id - ELECTRODE_START_ID} at frame {s1_idx}")
                
                self.update_3d_positions()
                
            elif k == 32 and yolo is not None:
                res = yolo.predict(img, conf=CONFIG["yolo_conf"], verbose=False)
                if res and res[0].boxes is not None:
                    added = 0
                    merged = 0
                    rejected_low_conf = 0
                    confidences = res[0].boxes.conf.cpu().numpy()
                    
                    for i, box in enumerate(res[0].boxes.xyxy.cpu().numpy()):
                        confidence = confidences[i]
                        if confidence < CONFIG["yolo_conf_accept"]:
                            rejected_low_conf += 1
                            continue
                        
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        
                        # Check if near existing 2D annotation in this frame
                        is_dup = False
                        for obj_id in self.annotations_2d:
                            if obj_id < ELECTRODE_START_ID:
                                continue
                            if s1_idx in self.annotations_2d[obj_id]:
                                eu, ev = self.annotations_2d[obj_id][s1_idx]
                                if np.sqrt((cx - eu)**2 + (cy - ev)**2) < 25:
                                    is_dup = True
                                    break
                        
                        if is_dup:
                            continue
                        
                        # Check 3D distance first (handles different viewpoints)
                        closest_id, closest_dist_3d = self.find_closest_electrode_3d(cx, cy, s1_idx)
                        if closest_id is not None:
                            self.annotations_2d[closest_id][s1_idx] = (float(cx), float(cy))
                            merged += 1
                            continue
                        
                        # Fall back to 2D projection matching
                        closest_id, closest_dist_2d = self.find_closest_projected_electrode(cx, cy, s1_idx)
                        if closest_id is not None:
                            self.annotations_2d[closest_id][s1_idx] = (float(cx), float(cy))
                            merged += 1
                        else:
                            # Create new electrode
                            new_id = self.next_electrode_id
                            self.next_electrode_id += 1
                            self.annotations_2d[new_id] = {s1_idx: (float(cx), float(cy))}
                            added += 1
                    
                    msg = f"  Auto-detected: {added} new"
                    if merged > 0:
                        msg += f", {merged} merged with existing"
                    if rejected_low_conf > 0:
                        msg += f" (rejected {rejected_low_conf} low-conf <{CONFIG['yolo_conf_accept']*100:.0f}%)"
                    print(msg)
                    self.update_3d_positions()
            elif k == ord('q'):
                break
        
        cv2.destroyWindow("Electrodes")
        self.update_3d_positions()
        n_electrodes = len([k for k in self.positions_3d if k >= ELECTRODE_START_ID])
        print(f"\n[OK] {n_electrodes} electrodes with 3D positions")
    
    def _annotate_single_object(self, obj_id, name, is_landmark=True):
        print(f"\n  --- Annotate {name} ---")
        print(f"  Click on {name} in multiple frames for better accuracy")
        print(f"  Y: Done | R: Clear all | A/D: Navigate")
        self.annotations_2d[obj_id] = {}
        vggt_frames = sorted(self.vggt_data["s1_to_vggt"].keys())
        idx = len(vggt_frames) // 2
        color = LANDMARK_COLORS.get(obj_id, (0, 255, 255))
        cv2.namedWindow(f"Annotate {name}")
        click_pos = None
        
        def mouse(e, x, y, f, p):
            nonlocal click_pos
            if e == cv2.EVENT_LBUTTONDOWN:
                click_pos = (int(x / self.scale), int(y / self.scale))
        
        cv2.setMouseCallback(f"Annotate {name}", mouse)
        
        while True:
            s1_idx = vggt_frames[idx]
            frame_path = os.path.join(FRAME_DIR, self.frames[s1_idx])
            img = cv2.imread(frame_path)
            disp, _ = resize_for_display(img, CONFIG["display_height"])
            
            for ann_frame, (u, v) in self.annotations_2d[obj_id].items():
                if ann_frame == s1_idx:
                    dx, dy = int(u * self.scale), int(v * self.scale)
                    cv2.circle(disp, (dx, dy), 10, color, -1)
                    cv2.circle(disp, (dx, dy), 12, (255, 255, 255), 2)
            
            if len(self.annotations_2d[obj_id]) >= 1:
                p3d = self.triangulate_point(obj_id)
                if p3d is not None:
                    vggt_idx = self.get_vggt_frame_idx(s1_idx)
                    if vggt_idx is not None:
                        proj, in_front = project_3d_to_2d(
                            p3d,
                            self.vggt_data["intrinsics"][vggt_idx],
                            self.vggt_data["extrinsics"][vggt_idx]
                        )
                        if in_front:
                            u_s1, v_s1 = vggt_to_script1_coords(proj[0], proj[1], 
                                                                  self.crop_info["w"], self.crop_info["h"])
                            dx, dy = int(u_s1 * self.scale), int(v_s1 * self.scale)
                            cv2.drawMarker(disp, (dx, dy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            n_obs = len(self.annotations_2d[obj_id])
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(disp, f"{name} | Frame {s1_idx} | Observations: {n_obs}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(disp, "Click to mark | A/D:Nav | Y:Done | R:Clear", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow(f"Annotate {name}", disp)
            
            k = cv2.waitKey(1) & 0xFF
            
            if k in [ord('a'), 81]: 
                idx = max(0, idx - 1)
            elif k in [ord('d'), 83]: 
                idx = min(len(vggt_frames) - 1, idx + 1)
            elif k == ord('w'): 
                idx = max(0, idx - 5)
            elif k == ord('s'): 
                idx = min(len(vggt_frames) - 1, idx + 5)
            elif click_pos is not None:
                u, v = click_pos
                click_pos = None
                removed = False
                if s1_idx in self.annotations_2d[obj_id]:
                    eu, ev = self.annotations_2d[obj_id][s1_idx]
                    if np.sqrt((u - eu)**2 + (v - ev)**2) < 20:
                        del self.annotations_2d[obj_id][s1_idx]
                        removed = True
                        print(f"    Removed annotation from frame {s1_idx}")
                if not removed:
                    self.annotations_2d[obj_id][s1_idx] = (u, v)
                    print(f"    Added annotation at frame {s1_idx}: ({u:.0f}, {v:.0f})")
            elif k == ord('r'):
                self.annotations_2d[obj_id] = {}
                print(f"    Cleared all annotations for {name}")
            elif k == ord('y'):
                if len(self.annotations_2d[obj_id]) >= 1:
                    break
                else:
                    print(f"    Need at least 1 annotation!")
            elif k == ord('q'):
                sys.exit(0)
        
        cv2.destroyWindow(f"Annotate {name}")


# PROJECTION-BASED TRACKING


def generate_tracking_from_3d(frames, vggt_data, crop_info, positions_3d):
    """
    Generate 2D tracking data by projecting 3D positions to each frame.
    Used for review visualization.
    """
    print("\n=== GENERATING TRACKING FROM 3D PROJECTIONS ===")
    
    tracking = {}
    
    img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    img_h, img_w = img0.shape[:2]
    
    for s1_idx in tqdm(range(len(frames)), desc="Projecting"):
        vggt_idx = vggt_data["s1_to_vggt"].get(s1_idx, None)
        
        if vggt_idx is None:
            # Interpolate for frames without VGGT data
            s1_indices = sorted(vggt_data["s1_to_vggt"].keys())
            if not s1_indices:
                continue
            
            before = [i for i in s1_indices if i < s1_idx]
            after = [i for i in s1_indices if i > s1_idx]
            
            if before and after:
                b_idx = before[-1]
                a_idx = after[0]
                t = (s1_idx - b_idx) / (a_idx - b_idx)
                b_vggt = vggt_data["s1_to_vggt"][b_idx]
                a_vggt = vggt_data["s1_to_vggt"][a_idx]
                intrinsic = vggt_data["intrinsics"][b_vggt]
                extrinsic = (1 - t) * vggt_data["extrinsics"][b_vggt] + t * vggt_data["extrinsics"][a_vggt]
            elif before:
                b_vggt = vggt_data["s1_to_vggt"][before[-1]]
                intrinsic = vggt_data["intrinsics"][b_vggt]
                extrinsic = vggt_data["extrinsics"][b_vggt]
            elif after:
                a_vggt = vggt_data["s1_to_vggt"][after[0]]
                intrinsic = vggt_data["intrinsics"][a_vggt]
                extrinsic = vggt_data["extrinsics"][a_vggt]
            else:
                continue
        else:
            intrinsic = vggt_data["intrinsics"][vggt_idx]
            extrinsic = vggt_data["extrinsics"][vggt_idx]
        
        frame_tracking = {}
        
        for obj_id, pos_3d in positions_3d.items():
            # Project to 2D
            proj, in_front = project_3d_to_2d(pos_3d, intrinsic, extrinsic)
            
            if not in_front:
                continue
            
            u_vggt, v_vggt = proj
            u_s1, v_s1 = vggt_to_script1_coords(u_vggt, v_vggt, crop_info["w"], crop_info["h"])
            
            # Check bounds
            if not (0 <= u_s1 < img_w and 0 <= v_s1 < img_h):
                continue
            
            frame_tracking[obj_id] = (u_s1, v_s1)
        
        if frame_tracking:
            tracking[s1_idx] = frame_tracking
    
    print(f"  Generated tracking for {len(tracking)} frames")
    
    return tracking


# REVIEW


def review_tracking(frames, tracking, positions_3d, vggt_data, crop_info):
    print("\n=== REVIEW TRACKING ===")
    idx = 0
    playing = False
    last_t = time.time()
    img0 = cv2.imread(os.path.join(FRAME_DIR, frames[0]))
    _, scale = resize_for_display(img0, CONFIG["display_height"])
    cv2.namedWindow("Review")
    
    while True:
        if playing and time.time() - last_t > 1.0 / CONFIG["playback_fps"]:
            idx = (idx + 1) % len(frames)
            last_t = time.time()
        
        img = cv2.imread(os.path.join(FRAME_DIR, frames[idx]))
        disp, _ = resize_for_display(img, CONFIG["display_height"])
        
        if idx in tracking:
            for obj_id, (u, v) in tracking[idx].items():
                color = get_color(obj_id)
                dx, dy = int(u * scale), int(v * scale)
                cv2.circle(disp, (dx, dy), 6, color, -1)
                cv2.circle(disp, (dx, dy), 8, (255, 255, 255), 1)
                if obj_id < NUM_LANDMARKS:
                    label = LANDMARK_SHORT.get(obj_id, f"L{obj_id}")
                else:
                    label = f"E{obj_id - ELECTRODE_START_ID}"
                cv2.putText(disp, label, (dx + 10, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        n_tracked = len(tracking.get(idx, {}))
        has_vggt = idx in vggt_data["s1_to_vggt"]
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 50), (0, 0, 0), -1)
        status = "PLAY" if playing else "PAUSE"
        vggt_status = "VGGT" if has_vggt else "interp"
        cv2.putText(disp, f"Frame {idx}/{len(frames)-1} | {status} | Tracked: {n_tracked} | {vggt_status}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, "A/D:Nav | P:Play | SPACE:Accept | R:Redo", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.imshow("Review", disp)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k in [ord('a'), 81] and not playing: idx = max(0, idx - 1)
        elif k in [ord('d'), 83] and not playing: idx = min(len(frames) - 1, idx + 1)
        elif k == ord('w') and not playing: idx = max(0, idx - 10)
        elif k == ord('s') and not playing: idx = min(len(frames) - 1, idx + 10)
        elif k == ord('p'): playing = not playing
        elif k == 32:
            cv2.destroyWindow("Review")
            return True
        elif k == ord('r'):
            cv2.destroyWindow("Review")
            return False
        elif k == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


# SAVE RESULTS


def save_results(tracking, positions_3d, output_paths):
    """Save tracking and 3D positions to video-specific directory."""
    print("\n=== SAVING RESULTS ===")
    
    tracking_file = output_paths["tracking_file"]
    points_3d_file = output_paths["points_3d_file"]
    
    with open(tracking_file, "wb") as f:
        pickle.dump(tracking, f)
    print(f"  {tracking_file}")
    
    with open(points_3d_file, "wb") as f:
        pickle.dump(positions_3d, f)
    print(f"  {points_3d_file}")
    
    # Statistics
    n_landmarks = sum(1 for i in positions_3d.keys() if isinstance(i, int) and i < NUM_LANDMARKS)
    n_electrodes = sum(1 for i in positions_3d.keys() if isinstance(i, int) and i >= ELECTRODE_START_ID)
    
    print(f"\n  Landmarks: {n_landmarks}")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Total 3D points: {len(positions_3d)}")


# MAIN


def main():
    video = select_video()
    
    # Setup video-specific output directory
    output_paths = setup_video_output_dir(video)

    print("EEG ELECTRODE REGISTRATION")
    print("Projection-Based Tracking with Quality Filtering")
    
    # Step 1: Extract frames
    frames, crop_info = extract_frames(video, output_paths)
    
    # Step 2: Run VGGT
    vggt_data = run_vggt_full(frames, crop_info, output_paths)
    
    # Step 3: Load YOLO
    yolo = load_yolo()
    
    while True:
        # Step 4: Multi-view annotation
        annotator = MultiViewAnnotator(frames, vggt_data, crop_info)
        annotator.annotate_landmarks()
        annotator.annotate_electrodes(yolo)
        
        # Step 5: Generate tracking for review
        tracking = generate_tracking_from_3d(
            frames, vggt_data, crop_info, annotator.positions_3d
        )
        
        # Step 6: Review
        if review_tracking(frames, tracking, annotator.positions_3d, vggt_data, crop_info):
            break
        print("\n  Restarting annotation...\n")
    
    # Step 7: Save
    save_results(tracking, annotator.positions_3d, output_paths)
    
    print("SCRIPT 1 COMPLETE!")
    print(f"Results saved to: {output_paths['results_dir']}")
    print(f"Next: python script2.py")


if __name__ == "__main__":
    main()