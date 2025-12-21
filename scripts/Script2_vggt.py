"""
VGGT 3D Head Reconstruction (Step 2)
====================================
Requirements:
1. You MUST run Script 1 first to extract frames and crop info.
2. This script uses those exact frames to ensure 3D alignment.
"""

import os
import sys
import glob
import time
import shutil
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from contextlib import nullcontext


# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.join(BASE_DIR, "vggt")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "vggt_output")

# Input: The frames created by Script 1
SCRIPT1_FRAMES_DIR = os.path.join(BASE_DIR, "frames")
CROP_INFO_FILE = os.path.join(BASE_DIR, "results", "crop_info.json")

# Temporary folder for VGGT (resized 518x518 images)
TEMP_VGGT_DIR = os.path.join(DATA_DIR, "vggt_ready_frames")

# Settings
MAX_FRAMES_FOR_3D = 12  # VGGT works best with ~50 frames
TARGET_SIZE = 518

sys.path.insert(0, REPO_PATH)


# SETUP


def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.geometry import (
            unproject_depth_map_to_point_map,
            closed_form_inverse_se3
        )
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        print("VGGT modules imported")
        return VGGT, load_and_preprocess_images, unproject_depth_map_to_point_map, closed_form_inverse_se3, pose_encoding_to_extri_intri
    except ImportError as e:
        print(f" CRITICAL ERROR: Could not import VGGT.\n   {e}")
        sys.exit(1)

VGGT, load_and_preprocess_images, unproject_depth_map_to_point_map, closed_form_inverse_se3, pose_encoding_to_extri_intri = setup_environment()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f" Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print(" Using CPU. This may be slow.")


# 1. LOAD DATA FROM SCRIPT 1


def load_script1_data():
    print(f"\n--- 1. Loading Data from Script 1 ---")
    
    # Check for frames
    if not os.path.exists(SCRIPT1_FRAMES_DIR):
        print(f" Error: Frames folder not found: {SCRIPT1_FRAMES_DIR}")
        print(" Please run Script 1 (Tracking) first!")
        sys.exit(1)

    # Check for crop info
    if not os.path.exists(CROP_INFO_FILE):
        print(f" Error: Crop info not found: {CROP_INFO_FILE}")
        print(" Please run Script 1 (Tracking) first!")
        sys.exit(1)
        
    # Load frames
    all_frames = sorted(glob.glob(os.path.join(SCRIPT1_FRAMES_DIR, "*.jpg")))
    if len(all_frames) == 0:
        print(" Error: Frames folder is empty.")
        sys.exit(1)
        
    print(f"  Found {len(all_frames)} frames from tracking step.")
    
    # Subsample frames if we have too many (VGGT can be slow with >50)
    # We select indices evenly: [0, 5, 10, 15...]
    num_select = min(len(all_frames), MAX_FRAMES_FOR_3D)
    indices = np.linspace(0, len(all_frames)-1, num_select, dtype=int)
    
    selected_paths = [all_frames[i] for i in indices]
    
    # Create the mapping: VGGT Index -> Script 1 Frame Index
    # We parse the filename "00042.jpg" -> 42 to get the original Script 1 index
    script1_indices = []
    for p in selected_paths:
        fname = os.path.basename(p)
        idx = int(os.path.splitext(fname)[0])
        script1_indices.append(idx)
        
    print(f"  Selected {len(selected_paths)} frames for 3D reconstruction.")
    
    return selected_paths, script1_indices


# 2. PREPROCESS FOR VGGT


def preprocess_frames(frame_paths):
    print(f"\n--- 2. Preprocessing for VGGT ({TARGET_SIZE}x{TARGET_SIZE}) ---")
    
    if os.path.exists(TEMP_VGGT_DIR): shutil.rmtree(TEMP_VGGT_DIR)
    os.makedirs(TEMP_VGGT_DIR, exist_ok=True)
    
    processed_paths = []
    
    for i, path in enumerate(tqdm(frame_paths, desc="Resizing")):
        img = cv2.imread(path)
        if img is None: continue
        
        h, w = img.shape[:2]
        scale = TARGET_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to square (black bars)
        pad_w = (TARGET_SIZE - new_w) // 2
        pad_h = (TARGET_SIZE - new_h) // 2
        img = cv2.copyMakeBorder(
            img, pad_h, TARGET_SIZE - new_h - pad_h,
            pad_w, TARGET_SIZE - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        out_path = os.path.join(TEMP_VGGT_DIR, f"{i:05d}.jpg")
        cv2.imwrite(out_path, img)
        processed_paths.append(out_path)
        
    return processed_paths

# 3. RUN AI MODEL


def run_vggt(image_paths):
    print(f"\n--- 3. Running VGGT Inference ---")
    
    # Load Model
    checkpoint = os.path.join(BASE_DIR, "checkpoints", "vggt_model.pt")
    model = VGGT()
    
    if not os.path.exists(checkpoint):
        print("  Downloading model weights (4GB)...")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        
        # progress=True adds the download bar!
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=DEVICE,
            progress=True 
        )
        torch.save(state, checkpoint)
    
    print("  Loading model from disk (4GB)... please wait.", end="", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(" Done!")
    
    # Inference
    print("  Processing on device... (This may take a while)", end="", flush=True)
    images = load_and_preprocess_images(image_paths).to(DEVICE)
    
    if DEVICE.type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
        
    start = time.time()
    with torch.no_grad(), ctx:
        preds = model(images)
    print(f" Done ({time.time()-start:.1f}s)")
    
    # Post-process
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    
    def cpu(x): return x.float().cpu().numpy()
    
    return {
        "images": cpu(images),
        "depth": cpu(preds["depth"]),
        "depth_conf": cpu(preds["depth_conf"]),
        "extrinsic": cpu(extrinsic),
        "intrinsic": cpu(intrinsic)
    }


# 4. SAVE RESULTS


def save_data(results, script1_indices):
    print(f"\n--- 4. Saving Results ---")
    
    # Compute Point Cloud
    world_points = unproject_depth_map_to_point_map(
        results["depth"], results["extrinsic"], results["intrinsic"]
    )
    
    # Reshape for saving
    points = world_points.reshape(-1, 3)
    colors = (results["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    conf = results["depth_conf"].reshape(-1)
    
    # Filter bad points
    valid = np.isfinite(points).all(axis=1)
    points, colors, conf = points[valid], colors[valid], conf[valid]
    
    # Center
    center = np.median(points, axis=0)
    points -= center
    
    # Save NPZ
    save_path = os.path.join(OUTPUT_DIR, "reconstruction.npz")
    np.savez_compressed(
        save_path,
        # 1. Visualization Data
        points=points, colors=colors, confidence=conf, center=center,
        
        # 2. Math Data (for Bridge)
        images=results["images"],
        depth=results["depth"],
        depth_conf=results["depth_conf"],
        extrinsics=results["extrinsic"],
        intrinsics=results["intrinsic"],
        
        # 3. Explicit Frame Mapping (VGGT Index -> Script 1 Index)
        frame_mapping_keys=np.arange(len(script1_indices)), # VGGT indices (0, 1, 2...)
        frame_mapping_values=np.array(script1_indices)      # Script 1 indices (0, 10, 20...)
    )
    
    print(f" Saved: {save_path}")
    return points, colors, conf


# 5. VIEWER


def launch_viewer(points, colors, conf):
    try:
        import viser
    except ImportError:
        print(" Viser not installed. Skipping viewer.")
        return

    port = 8080
    print(f"\n--- Starting Viewer on http://localhost:{port} ---")
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    
    # Controls
    with server.gui.add_folder("Settings"):
        conf_slider = server.gui.add_slider("Confidence", 0, 99, 1, 30)
        size_slider = server.gui.add_slider("Point Size", 0.001, 0.015, 0.001, 0.005)
        
    with server.gui.add_folder("Cropping"):
        cx = server.gui.add_slider("Clip X", 0.1, 3.0, 0.05, 2.0)
        cy = server.gui.add_slider("Clip Y", 0.1, 3.0, 0.05, 2.0)
        cz = server.gui.add_slider("Clip Z", 0.1, 3.0, 0.05, 2.0)
        
    pcd = server.scene.add_point_cloud("head", points, colors, size_slider.value)
    
    def update(_):
        mask = (conf >= np.percentile(conf, conf_slider.value)) & \
               (np.abs(points[:,0]) < cx.value) & \
               (np.abs(points[:,1]) < cy.value) & \
               (np.abs(points[:,2]) < cz.value)
        pcd.points = points[mask]
        pcd.colors = colors[mask]
        pcd.point_size = size_slider.value
        
    for s in [conf_slider, size_slider, cx, cy, cz]: s.on_update(update)
    update(None)
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")


# MAIN


def main():
    print("=" * 70)
    print("VGGT 3D RECONSTRUCTION (PIPELINE STEP 2)")
    print("=" * 70)
    
    # 1. Get Data
    paths, indices = load_script1_data()
    
    # 2. Preprocess
    ready_paths = preprocess_frames(paths)
    
    # 3. AI
    results = run_vggt(ready_paths)
    
    # 4. Save
    pts, cols, cnf = save_data(results, indices)
    
    # 5. View
    launch_viewer(pts, cols, cnf)

if __name__ == "__main__":
    main()