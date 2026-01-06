"""

SCRIPT 3: 2D→3D PROJECTION WITH FRAME ALIGNMENT


PURPOSE:
    Project 2D tracking data to 3D with proper geometric calculations.
    
KEY FIX:
    Does NOT require all 3 landmarks in a single frame!
    Instead, we:
    1. Collect landmark observations from ALL frames
    2. Build a "virtual reference" from averaged landmarks
    3. Align each frame using whatever landmarks ARE visible
    
INPUT:
    - tracking_results.pkl (from Script 1)
    - reconstruction.npz (from Script 2)
    - crop_info.json (from Script 1)
    
OUTPUT:
    - electrodes_3d.json: Final electrode positions in head coordinates (mm)
    - electrodes_3d.ply: For 3D visualization

COORDINATE SYSTEM:
    Origin: Midpoint between LPA and RPA (ears)
    X-axis: LPA → RPA (left to right)
    Y-axis: INION → NAS (back to front)
    Z-axis: Down → Up (perpendicular to X and Y)

"""

import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm


# CONFIGURATION


print("=" * 70)
print("SCRIPT 3: 2D→3D PROJECTION WITH FRAME ALIGNMENT")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
CROP_FILE = os.path.join(RESULTS_DIR, "crop_info.json")

OUTPUT_JSON = os.path.join(RESULTS_DIR, "electrodes_3d.json")
OUTPUT_PLY = os.path.join(RESULTS_DIR, "electrodes_3d.ply")

# Landmark IDs (from Script 1)
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3

LANDMARK_NAMES = {
    LANDMARK_NAS: "NAS",
    LANDMARK_LPA: "LPA", 
    LANDMARK_RPA: "RPA"
}

# Measurement conversion factors
ARC_TO_CHORD = 0.92
CIRCUMFERENCE_TO_EAR = 0.26

# Alignment settings
MIN_LANDMARKS_FOR_ALIGNMENT = 2  # Minimum landmarks needed per frame

print("=" * 70)


# ==============================================================================
# COORDINATE TRANSFORMATION
# ==============================================================================

def script1_to_vggt_coords(u, v, crop_w, crop_h, vggt_size=518):
    """Convert Script 1 pixel coordinates to VGGT coordinates."""
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


def unproject_pixel(u, v, depth_map, intrinsic, extrinsic):
    """Convert 2D pixel to 3D world point."""
    H, W = depth_map.shape
    
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    
    # Bilinear interpolation
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
    return (np.linalg.inv(extrinsic) @ P_cam)[:3]


# ==============================================================================
# ROBUST AVERAGING
# ==============================================================================

def robust_average(points, outlier_std=2.0):
    """
    Compute robust average with outlier removal.
    
    Args:
        points: list of 3D points
        outlier_std: remove points > N std from median
    
    Returns:
        averaged point, or None if insufficient data
    """
    if len(points) < 1:
        return None
    
    pts = np.array(points)
    
    if len(pts) == 1:
        return pts[0]
    
    # Use median as robust center
    median = np.median(pts, axis=0)
    
    if len(pts) < 3:
        return median
    
    # Remove outliers
    dists = np.linalg.norm(pts - median, axis=1)
    threshold = np.mean(dists) + outlier_std * np.std(dists)
    mask = dists < threshold
    
    if np.sum(mask) < 1:
        return median
    
    return np.mean(pts[mask], axis=0)


# ==============================================================================
# PROCRUSTES ALIGNMENT
# ==============================================================================

def compute_procrustes_transform(source_points, target_points):
    """
    Compute rigid transformation (rotation + translation + scale).
    
    Can work with 2 or 3 points (2 points = less accurate but still useful).
    """
    n_points = len(source_points)
    
    if n_points < 2:
        return None
    
    source = np.array(source_points)
    target = np.array(target_points)
    
    # Center both point sets
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    
    source_centered = source - source_center
    target_centered = target - target_center
    
    # Compute scale
    source_scale = np.sqrt(np.sum(source_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))
    
    if source_scale < 1e-10 or target_scale < 1e-10:
        return None
    
    # Normalize
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale
    
    # Compute optimal rotation using SVD
    H = target_normalized.T @ source_normalized
    U, S, Vt = np.linalg.svd(H)
    
    R = U @ Vt
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    scale = target_scale / source_scale
    translation = target_center - scale * (R @ source_center)
    
    return {
        "rotation": R,
        "translation": translation,
        "scale": scale
    }


def apply_procrustes(point, transform):
    """Apply Procrustes transformation to a point."""
    return transform["scale"] * (transform["rotation"] @ point) + transform["translation"]


# ==============================================================================
# 3D INION ESTIMATION
# ==============================================================================

def estimate_inion_3d(nas, lpa, rpa):
    """
    Estimate INION position in 3D from NAS, LPA, RPA.
    Done in 3D to avoid perspective distortion.
    """
    origin = (lpa + rpa) / 2.0
    
    ear_axis = rpa - lpa
    ear_len = np.linalg.norm(ear_axis)
    if ear_len < 1e-6:
        return None
    ear_axis = ear_axis / ear_len
    
    nas_vec = nas - origin
    forward = nas_vec - np.dot(nas_vec, ear_axis) * ear_axis
    forward_len = np.linalg.norm(forward)
    
    if forward_len < 1e-6:
        return None
    
    forward = forward / forward_len
    return origin - forward * forward_len


# ==============================================================================
# HEAD COORDINATE SYSTEM
# ==============================================================================

def build_head_transform(nas, lpa, rpa, measured_mm):
    """Build transformation from world space to head-centered mm space."""
    
    inion = estimate_inion_3d(nas, lpa, rpa)
    if inion is None:
        print("  ERROR: Could not estimate INION")
        return None
    
    origin = (lpa + rpa) / 2.0
    
    x_axis = rpa - lpa
    raw_ear_dist = np.linalg.norm(x_axis)
    x_axis = x_axis / raw_ear_dist
    
    y_vec = nas - inion
    y_axis = y_vec - np.dot(y_vec, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    R = np.array([x_axis, y_axis, z_axis])
    scale = measured_mm / raw_ear_dist
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "raw_ear_dist": raw_ear_dist,
        "inion": inion
    }


def apply_head_transform(point, transform):
    """Transform point to head coordinates (mm)."""
    centered = point - transform["origin"]
    rotated = transform["rotation"] @ centered
    scaled = rotated * transform["scale"]
    return scaled


# ==============================================================================
# MEASUREMENT INPUT
# ==============================================================================

def get_measurement():
    """Get ear-to-ear measurement from user."""
    
    print("\n" + "=" * 60)
    print("HEAD MEASUREMENT")
    print("=" * 60)
    print("[1] Caliper (direct ear-to-ear, tragus to tragus)")
    print("[2] Tape arc (over top of head, ear to ear)")
    print("[3] Head circumference")
    print("[4] Default (150mm)")
    print()
    
    choice = input("Choice: ").strip()
    
    if choice == "1":
        mm = float(input("Enter ear-to-ear distance (mm): "))
        return mm, "caliper"
    elif choice == "2":
        arc = float(input("Enter arc measurement (mm): "))
        chord = arc * ARC_TO_CHORD
        print(f"  → Ear-to-ear (chord): {chord:.1f} mm")
        return chord, "arc"
    elif choice == "3":
        circ = float(input("Enter circumference (mm): "))
        chord = circ * CIRCUMFERENCE_TO_EAR
        print(f"  → Ear-to-ear: {chord:.1f} mm")
        return chord, "circumference"
    else:
        print("  → Using default: 150 mm")
        return 150.0, "default"


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    print("\n--- Step 1: Loading Data ---")
    
    for f, name in [(TRACKING_FILE, "Tracking"), (RECON_FILE, "Reconstruction"), (CROP_FILE, "Crop info")]:
        if not os.path.exists(f):
            print(f"  ERROR: {name} not found: {f}")
            sys.exit(1)
    
    with open(TRACKING_FILE, "rb") as f:
        tracking = pickle.load(f)
    
    with open(CROP_FILE, "r") as f:
        crop = json.load(f)
    
    recon = np.load(RECON_FILE)
    
    print(f"  Tracking: {len(tracking)} frames")
    print(f"  Crop: {crop['w']}x{crop['h']}")
    
    # ==========================================================================
    # STEP 2: BUILD FRAME MAPPING
    # ==========================================================================
    print("\n--- Step 2: Frame Mapping ---")
    
    vggt_indices = recon["frame_mapping_keys"]
    s1_indices = recon["frame_mapping_values"]
    s1_to_vggt = {int(s1): int(vggt) for vggt, s1 in zip(vggt_indices, s1_indices)}
    
    matched = sum(1 for s1 in tracking.keys() if s1 in s1_to_vggt)
    print(f"  VGGT frames: {len(vggt_indices)}")
    print(f"  Tracking frames with depth: {matched}")
    
    depths = np.squeeze(recon["depth"])
    intrinsics = np.squeeze(recon["intrinsics"])
    extrinsics = np.squeeze(recon["extrinsics"])
    
    # ==========================================================================
    # STEP 3: FIRST PASS - UNPROJECT ALL FRAMES (NO ALIGNMENT YET)
    # ==========================================================================
    print("\n--- Step 3: First Pass - Unprojecting to 3D ---")
    
    all_frames_3d = {}  # {frame_idx: {obj_id: 3D_point}}
    
    for s1_idx, frame_tracks in tqdm(tracking.items(), desc="  Processing"):
        if s1_idx not in s1_to_vggt:
            continue
        
        vggt_idx = s1_to_vggt[s1_idx]
        frame_3d = {}
        
        for obj_id, (u, v) in frame_tracks.items():
            u_vggt, v_vggt = script1_to_vggt_coords(u, v, crop["w"], crop["h"])
            p3d = unproject_pixel(u_vggt, v_vggt, depths[vggt_idx], 
                                  intrinsics[vggt_idx], extrinsics[vggt_idx])
            if p3d is not None:
                frame_3d[obj_id] = p3d
        
        if frame_3d:
            all_frames_3d[s1_idx] = frame_3d
    
    print(f"  Frames unprojected: {len(all_frames_3d)}")
    
    # ==========================================================================
    # STEP 4: BUILD VIRTUAL REFERENCE FROM ALL LANDMARK OBSERVATIONS
    # ==========================================================================
    print("\n--- Step 4: Building Virtual Reference ---")
    print("  (Collecting landmarks from ALL frames, not just one)")
    
    # Collect all observations of each landmark
    landmark_observations = {lid: [] for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]}
    
    for frame_idx, frame_3d in all_frames_3d.items():
        for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            if lid in frame_3d:
                landmark_observations[lid].append(frame_3d[lid])
    
    # Report observations
    for lid, name in LANDMARK_NAMES.items():
        n_obs = len(landmark_observations[lid])
        print(f"  {name}: {n_obs} observations across all frames")
    
    # Check we have enough observations
    missing = [LANDMARK_NAMES[lid] for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA] 
               if len(landmark_observations[lid]) == 0]
    
    if missing:
        print(f"\n  ERROR: No observations for: {', '.join(missing)}")
        print("  → Check that landmarks were clicked in Script 1")
        print("  → Use DIFFERENT colored stickers!")
        sys.exit(1)
    
    # Compute reference landmarks (robust average of all observations)
    ref_landmarks = {}
    for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
        ref_landmarks[lid] = robust_average(landmark_observations[lid])
        pos = ref_landmarks[lid]
        print(f"  {LANDMARK_NAMES[lid]} reference: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # Sanity check
    ear_dist = np.linalg.norm(ref_landmarks[LANDMARK_RPA] - ref_landmarks[LANDMARK_LPA])
    nas_to_center = np.linalg.norm(ref_landmarks[LANDMARK_NAS] - 
                                   (ref_landmarks[LANDMARK_LPA] + ref_landmarks[LANDMARK_RPA]) / 2)
    
    print(f"\n  Reference distances (VGGT units):")
    print(f"    LPA-RPA: {ear_dist:.4f}")
    print(f"    NAS to ear-center: {nas_to_center:.4f}")
    
    if ear_dist < 0.01:
        print("\n  ⚠️ WARNING: LPA-RPA distance is very small!")
        print("     Landmarks may be tracked as the same point.")
        print("     → Use DIFFERENT colored stickers!")
    
    # ==========================================================================
    # STEP 5: SECOND PASS - ALIGN EACH FRAME TO VIRTUAL REFERENCE
    # ==========================================================================
    print("\n--- Step 5: Second Pass - Aligning Frames ---")
    print(f"  (Using {MIN_LANDMARKS_FOR_ALIGNMENT}+ landmarks per frame)")
    
    aligned_frames = {}
    stats = {"aligned": 0, "skipped_few_landmarks": 0}
    
    for frame_idx, frame_3d in tqdm(all_frames_3d.items(), desc="  Aligning"):
        # Get landmarks visible in this frame
        frame_landmarks = {}
        for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]:
            if lid in frame_3d:
                frame_landmarks[lid] = frame_3d[lid]
        
        # Need at least MIN_LANDMARKS_FOR_ALIGNMENT landmarks
        if len(frame_landmarks) < MIN_LANDMARKS_FOR_ALIGNMENT:
            stats["skipped_few_landmarks"] += 1
            continue
        
        # Build source and target point arrays
        visible_lids = sorted(frame_landmarks.keys())
        source_pts = np.array([frame_landmarks[lid] for lid in visible_lids])
        target_pts = np.array([ref_landmarks[lid] for lid in visible_lids])
        
        # Compute Procrustes transform
        transform = compute_procrustes_transform(source_pts, target_pts)
        
        if transform is None:
            stats["skipped_few_landmarks"] += 1
            continue
        
        # Apply to all points in frame
        aligned = {}
        for obj_id, point in frame_3d.items():
            aligned[obj_id] = apply_procrustes(point, transform)
        
        aligned_frames[frame_idx] = aligned
        stats["aligned"] += 1
    
    print(f"\n  Frames aligned: {stats['aligned']}")
    print(f"  Frames skipped (< {MIN_LANDMARKS_FOR_ALIGNMENT} landmarks): {stats['skipped_few_landmarks']}")
    
    if stats["aligned"] == 0:
        print("\n  ERROR: No frames could be aligned!")
        print("  → Check landmark tracking quality")
        sys.exit(1)
    
    # ==========================================================================
    # STEP 6: AVERAGE ALIGNED POSITIONS
    # ==========================================================================
    print("\n--- Step 6: Averaging Aligned Positions ---")
    
    # Collect observations
    all_observations = {}
    for frame_idx, frame_data in aligned_frames.items():
        for obj_id, point in frame_data.items():
            if obj_id not in all_observations:
                all_observations[obj_id] = []
            all_observations[obj_id].append(point)
    
    # Average with outlier removal
    avg_points = {}
    
    for obj_id, observations in all_observations.items():
        avg = robust_average(observations)
        if avg is not None:
            avg_points[obj_id] = avg
            
            label = LANDMARK_NAMES.get(obj_id, f"E{obj_id - NUM_LANDMARKS}")
            print(f"    {label}: {len(observations)} obs → ({avg[0]:.4f}, {avg[1]:.4f}, {avg[2]:.4f})")
    
    print(f"\n  Total objects: {len(avg_points)}")
    
    # ==========================================================================
    # STEP 7: ALIGNMENT QUALITY CHECK
    # ==========================================================================
    print("\n--- Step 7: Alignment Quality ---")
    
    for lid, name in LANDMARK_NAMES.items():
        if lid in all_observations:
            pts = np.array(all_observations[lid])
            if len(pts) > 1:
                std = np.std(pts, axis=0)
                print(f"  {name} std dev: ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    
    # Verify landmarks exist
    if not all(lid in avg_points for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]):
        print("  ERROR: Missing landmarks after averaging!")
        sys.exit(1)
    
    # ==========================================================================
    # STEP 8: GET MEASUREMENT & BUILD TRANSFORM
    # ==========================================================================
    measured_mm, method = get_measurement()
    
    print("\n--- Step 8: Building Head Coordinate System ---")
    
    transform = build_head_transform(
        avg_points[LANDMARK_NAS],
        avg_points[LANDMARK_LPA],
        avg_points[LANDMARK_RPA],
        measured_mm
    )
    
    if transform is None:
        sys.exit(1)
    
    print(f"  Scale: {transform['scale']:.2f} mm/unit")
    print(f"  Raw ear-to-ear: {transform['raw_ear_dist']:.4f} units")
    
    # ==========================================================================
    # STEP 9: TRANSFORM TO HEAD COORDINATES
    # ==========================================================================
    print("\n--- Step 9: Transforming to mm ---")
    
    final_points = {}
    
    for obj_id, point in avg_points.items():
        final_points[obj_id] = apply_head_transform(point, transform)
    
    # Add estimated INION
    final_points["INION"] = apply_head_transform(transform["inion"], transform)
    
    # ==========================================================================
    # STEP 10: VERIFICATION
    # ==========================================================================
    print("\n--- Step 10: Verification ---")
    
    nas_f = final_points[LANDMARK_NAS]
    lpa_f = final_points[LANDMARK_LPA]
    rpa_f = final_points[LANDMARK_RPA]
    inion_f = final_points["INION"]
    
    print(f"\n  Landmark positions (mm):")
    print(f"    NAS:   ({nas_f[0]:7.1f}, {nas_f[1]:7.1f}, {nas_f[2]:7.1f})")
    print(f"    LPA:   ({lpa_f[0]:7.1f}, {lpa_f[1]:7.1f}, {lpa_f[2]:7.1f})")
    print(f"    RPA:   ({rpa_f[0]:7.1f}, {rpa_f[1]:7.1f}, {rpa_f[2]:7.1f})")
    print(f"    INION: ({inion_f[0]:7.1f}, {inion_f[1]:7.1f}, {inion_f[2]:7.1f})")
    
    final_ear = np.linalg.norm(rpa_f - lpa_f)
    final_nasi = np.linalg.norm(nas_f - inion_f)
    
    print(f"\n  Distances:")
    print(f"    Ear-to-ear: {final_ear:.1f} mm (target: {measured_mm:.1f})")
    print(f"    NAS-INION: {final_nasi:.1f} mm")
    
    print(f"\n  Expected positions (head coordinates):")
    print(f"    LPA: ({-measured_mm/2:.1f}, 0, 0)")
    print(f"    RPA: ({measured_mm/2:.1f}, 0, 0)")
    print(f"    NAS: (0, +Y, 0)")
    print(f"    INION: (0, -Y, 0)")
    
    # ==========================================================================
    # STEP 11: SAVE JSON
    # ==========================================================================
    print("\n--- Step 11: Saving Results ---")
    
    output = {
        "coordinate_system": {
            "origin": "midpoint between LPA and RPA",
            "x_axis": "left to right (LPA → RPA)",
            "y_axis": "back to front (INION → NAS)",
            "z_axis": "down to up",
        },
        "units": "mm",
        "measurement": {
            "method": method,
            "ear_to_ear_mm": measured_mm,
            "scale_factor": float(transform["scale"]),
        },
        "alignment": {
            "method": "virtual_reference",
            "description": "Reference built from averaged landmarks across all frames",
            "frames_aligned": stats["aligned"],
            "frames_skipped": stats["skipped_few_landmarks"],
            "min_landmarks_per_frame": MIN_LANDMARKS_FOR_ALIGNMENT,
        },
        "landmark_observations": {
            LANDMARK_NAMES[lid]: len(landmark_observations[lid]) 
            for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]
        },
        "landmarks": {},
        "electrodes": {}
    }
    
    for obj_id, pos in final_points.items():
        pos_list = pos.tolist()
        
        if obj_id == "INION":
            output["landmarks"]["INION"] = pos_list
        elif obj_id == LANDMARK_NAS:
            output["landmarks"]["NAS"] = pos_list
        elif obj_id == LANDMARK_LPA:
            output["landmarks"]["LPA"] = pos_list
        elif obj_id == LANDMARK_RPA:
            output["landmarks"]["RPA"] = pos_list
        else:
            output["electrodes"][f"E{obj_id - NUM_LANDMARKS}"] = pos_list
    
    output["num_electrodes"] = len(output["electrodes"])
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ {OUTPUT_JSON}")
    
    # ==========================================================================
    # STEP 12: SAVE PLY
    # ==========================================================================
    ply_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(final_points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    
    for obj_id, pos in final_points.items():
        if obj_id == "INION":
            r, g, b = 255, 165, 0   # Orange
        elif isinstance(obj_id, int) and obj_id < NUM_LANDMARKS:
            r, g, b = 255, 0, 0    # Red
        else:
            r, g, b = 0, 100, 255  # Blue
        
        ply_lines.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}")
    
    with open(OUTPUT_PLY, "w") as f:
        f.write("\n".join(ply_lines))
    
    print(f"✓ {OUTPUT_PLY}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SCRIPT 3 COMPLETE!")
    print("=" * 70)
    print(f"\n  Landmarks: {len(output['landmarks'])}")
    print(f"  Electrodes: {output['num_electrodes']}")
    print(f"  Measurement: {method} ({measured_mm:.1f} mm)")
    print(f"\n  Landmark observations:")
    for name, count in output["landmark_observations"].items():
        print(f"    {name}: {count} frames")
    print(f"\n  Frames aligned: {stats['aligned']}/{len(all_frames_3d)}")
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_PLY}")


if __name__ == "__main__":
    main()