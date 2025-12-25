"""
VGGT-YOLO Bridge Script (Step 3) - FIXED
========================================
"""

import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "electrodes_3d.json")
OUTPUT_PLY = os.path.join(RESULTS_DIR, "electrodes_3d.ply")

LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3

ARC_TO_CHORD_RATIO = 0.92
CIRCUMFERENCE_TO_EAR_RATIO = 0.26

# ==============================================================================
# MATH HELPERS
# ==============================================================================

def transform_coords_to_vggt_space(u, v, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    pad_w, pad_h = (vggt_size - new_w) // 2, (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h

def unproject_point(u, v, depth_map, intrinsic, extrinsic):
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
    return (np.linalg.inv(extrinsic) @ P_cam)[:3]

def estimate_inion_3d(nas, lpa, rpa):
    origin = (lpa + rpa) / 2.0
    ear_axis = rpa - lpa
    ear_axis /= np.linalg.norm(ear_axis)
    
    nas_vec = nas - origin
    forward_dir = nas_vec - np.dot(nas_vec, ear_axis) * ear_axis
    forward_len = np.linalg.norm(forward_dir)
    
    if forward_len < 1e-6:
        return None
    
    forward_dir /= forward_len
    return origin - forward_dir * forward_len

def define_final_transform(points_avg, measured_mm):
    NAS = points_avg[LANDMARK_NAS]
    LPA = points_avg[LANDMARK_LPA]
    RPA = points_avg[LANDMARK_RPA]
    
    INION = estimate_inion_3d(NAS, LPA, RPA)
    if INION is None:
        print("  ❌ Could not estimate INION")
        return None
    
    origin = (LPA + RPA) / 2.0
    
    x_axis = RPA - LPA
    raw_dist = np.linalg.norm(x_axis)
    x_axis /= raw_dist
    
    y_vec = NAS - INION
    y_axis = y_vec - np.dot(y_vec, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    R = np.array([x_axis, y_axis, z_axis])
    scale = measured_mm / raw_dist
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "raw_ear_dist": raw_dist,
        "est_inion": INION
    }

def apply_transform(p, t):
    return t["rotation"] @ (p - t["origin"]) * t["scale"]

# ==============================================================================
# MEASUREMENT
# ==============================================================================

def get_measurement():
    print("\n" + "="*60)
    print(" HEAD MEASUREMENT")
    print("="*60)
    print(" [1] Caliper (Ear-to-Ear direct)")
    print(" [2] Tape Arc (Over Head)")
    print(" [3] Circumference")
    print(" [4] Default (150mm)\n")
    
    c = input("Choice: ").strip()
    if c == "1": 
        return float(input("Enter mm: ")), "caliper"
    if c == "2": 
        arc = float(input("Enter arc mm: "))
        chord = arc * ARC_TO_CHORD_RATIO
        print(f"  → Chord: {chord:.1f} mm")
        return chord, "arc"
    if c == "3": 
        circ = float(input("Enter circumference mm: "))
        chord = circ * CIRCUMFERENCE_TO_EAR_RATIO
        print(f"  → Ear-to-ear: {chord:.1f} mm")
        return chord, "circumference"
    return 150.0, "default"

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("BRIDGE SCRIPT (STEP 3) - FIXED FRAME MAPPING")
    print("=" * 70)
    
    # ==========================================================================
    # 1. LOAD DATA
    # ==========================================================================
    print("\n--- 1. Loading Data ---")
    
    if not os.path.exists(TRACKING_FILE):
        print(f"  ❌ Missing: {TRACKING_FILE}")
        sys.exit(1)
    if not os.path.exists(RECON_FILE):
        print(f"  ❌ Missing: {RECON_FILE}")
        sys.exit(1)
        
    with open(TRACKING_FILE, "rb") as f: 
        tracking = pickle.load(f)
    with open(CROP_INFO_FILE, "r") as f: 
        crop = json.load(f)
    recon = np.load(RECON_FILE)
    
    # ==========================================================================
    # 2. BUILD FRAME MAPPING (FIXED!)
    # ==========================================================================
    print("\n--- 2. Building Frame Mapping ---")
    
    vggt_indices = recon["frame_mapping_keys"]    # [0, 1, 2, ..., 19]
    s1_indices = recon["frame_mapping_values"]     # [0, 13, 26, ..., 247]
    
    # CORRECT mapping: Script1 index → VGGT index
    s1_to_vggt = {int(s1): int(vggt) for vggt, s1 in zip(vggt_indices, s1_indices)}
    
    print(f"  VGGT frames: {len(vggt_indices)}")
    print(f"  Script1 frames in tracking: {len(tracking)}")
    print(f"  Mapping sample: S1[{s1_indices[0]}]→VGGT[0], S1[{s1_indices[-1]}]→VGGT[{vggt_indices[-1]}]")
    
    # Count how many tracking frames have VGGT depth
    matched_count = sum(1 for s1_idx in tracking.keys() if s1_idx in s1_to_vggt)
    print(f"  Tracking frames with VGGT depth: {matched_count}")
    
    # ==========================================================================
    # 3. LOAD DEPTH/CAMERA DATA
    # ==========================================================================
    depths = np.squeeze(recon["depth"])
    intrs = np.squeeze(recon["intrinsics"])
    extrs = np.squeeze(recon["extrinsics"])
    
    print(f"\n  Depth shape: {depths.shape}")
    print(f"  Intrinsics shape: {intrs.shape}")
    print(f"  Extrinsics shape: {extrs.shape}")
    
    # ==========================================================================
    # 4. COLLECT WORLD POINTS
    # ==========================================================================
    print("\n--- 3. Unprojecting to World Space ---")
    
    world_points = {}  # {eid: [list of 3D points]}
    frames_used = 0
    points_unprojected = 0
    
    for s1_idx, tracks in tqdm(tracking.items(), desc="  Processing"):
        if s1_idx not in s1_to_vggt:
            continue
        
        vggt_idx = s1_to_vggt[s1_idx]
        frames_used += 1
        
        for eid, (u, v) in tracks.items():
            u_v, v_v = transform_coords_to_vggt_space(u, v, crop["w"], crop["h"])
            p3d = unproject_point(u_v, v_v, depths[vggt_idx], intrs[vggt_idx], extrs[vggt_idx])
            
            if p3d is not None:
                if eid not in world_points:
                    world_points[eid] = []
                world_points[eid].append(p3d)
                points_unprojected += 1
    
    print(f"\n  Frames used: {frames_used}")
    print(f"  Points unprojected: {points_unprojected}")
    print(f"  Unique electrodes: {len(world_points)}")
    
    # ==========================================================================
    # 5. ROBUST AVERAGING
    # ==========================================================================
    print("\n--- 4. Averaging Positions ---")
    
    avg_points = {}
    skipped_few_obs = 0
    
    for eid, pts in world_points.items():
        arr = np.array(pts)
        
        if len(arr) < 2:
            skipped_few_obs += 1
            continue
        
        # Outlier removal
        mean = np.median(arr, axis=0)
        dists = np.linalg.norm(arr - mean, axis=1)
        threshold = np.mean(dists) + 2 * np.std(dists)
        mask = dists < threshold
        
        if np.sum(mask) < 2:
            avg_points[eid] = mean  # Use median if too few after filtering
        else:
            avg_points[eid] = np.mean(arr[mask], axis=0)
        
        # Print info
        label = {0: "NAS", 1: "LPA", 2: "RPA"}.get(eid, f"E{eid - NUM_LANDMARKS}")
        print(f"    {label}: {len(arr)} obs → ({avg_points[eid][0]:.4f}, {avg_points[eid][1]:.4f}, {avg_points[eid][2]:.4f})")
    
    print(f"\n  Electrodes with enough observations: {len(avg_points)}")
    print(f"  Skipped (< 2 observations): {skipped_few_obs}")
    
    # ==========================================================================
    # 6. CHECK LANDMARKS
    # ==========================================================================
    print("\n--- 5. Checking Landmarks ---")
    
    missing_landmarks = []
    for lid, name in [(LANDMARK_NAS, "NAS"), (LANDMARK_LPA, "LPA"), (LANDMARK_RPA, "RPA")]:
        if lid in avg_points:
            print(f"  ✓ {name} found")
        else:
            print(f"  ❌ {name} MISSING!")
            missing_landmarks.append(name)
    
    if missing_landmarks:
        print(f"\n  ❌ Cannot continue without landmarks: {', '.join(missing_landmarks)}")
        print("  → Check that landmarks are tracked correctly in Script 1")
        print("  → Use DIFFERENT colored stickers for NAS, LPA, RPA!")
        sys.exit(1)
    
    # Check landmark distances (sanity check)
    nas = avg_points[LANDMARK_NAS]
    lpa = avg_points[LANDMARK_LPA]
    rpa = avg_points[LANDMARK_RPA]
    
    ear_dist = np.linalg.norm(rpa - lpa)
    nas_to_center = np.linalg.norm(nas - (lpa + rpa) / 2)
    
    print(f"\n  Distances in VGGT units:")
    print(f"    LPA-RPA (ear-to-ear): {ear_dist:.4f}")
    print(f"    NAS to ear-center: {nas_to_center:.4f}")
    
    if ear_dist < 0.01:
        print(f"\n  ⚠️ WARNING: LPA-RPA distance is very small ({ear_dist:.6f})")
        print("     This suggests LPA and RPA are being tracked as the SAME point!")
        print("     → Use DIFFERENT colored stickers for landmarks!")
    
    # ==========================================================================
    # 7. GET MEASUREMENT & TRANSFORM
    # ==========================================================================
    mm, method = get_measurement()
    
    print(f"\n--- 6. Aligning & Scaling ---")
    
    transform = define_final_transform(avg_points, mm)
    if transform is None:
        sys.exit(1)
    
    print(f"  Scale factor: {transform['scale']:.2f} mm/unit")
    print(f"  Raw ear-to-ear: {transform['raw_ear_dist']:.4f} units")
    
    # Apply transform to all points
    final_pts = {}
    for eid, p in avg_points.items():
        final_pts[eid] = apply_transform(p, transform)
    
    # Add estimated INION
    final_pts["INION_EST"] = apply_transform(transform["est_inion"], transform)
    
    # ==========================================================================
    # 8. VERIFICATION
    # ==========================================================================
    print(f"\n--- 7. Verification ---")
    
    lpa_final = final_pts[LANDMARK_LPA]
    rpa_final = final_pts[LANDMARK_RPA]
    nas_final = final_pts[LANDMARK_NAS]
    inion_final = final_pts["INION_EST"]
    
    final_ear_dist = np.linalg.norm(rpa_final - lpa_final)
    final_nas_inion = np.linalg.norm(nas_final - inion_final)
    
    print(f"  Landmark positions (mm):")
    print(f"    NAS:   ({nas_final[0]:7.1f}, {nas_final[1]:7.1f}, {nas_final[2]:7.1f})")
    print(f"    LPA:   ({lpa_final[0]:7.1f}, {lpa_final[1]:7.1f}, {lpa_final[2]:7.1f})")
    print(f"    RPA:   ({rpa_final[0]:7.1f}, {rpa_final[1]:7.1f}, {rpa_final[2]:7.1f})")
    print(f"    INION: ({inion_final[0]:7.1f}, {inion_final[1]:7.1f}, {inion_final[2]:7.1f})")
    
    print(f"\n  Distances:")
    print(f"    Ear-to-ear (LPA↔RPA): {final_ear_dist:.1f} mm (should be {mm:.1f})")
    print(f"    NAS↔INION: {final_nas_inion:.1f} mm")
    
    print(f"\n  Expected landmark positions:")
    print(f"    LPA: ({-mm/2:.1f}, ~0, ~0) mm")
    print(f"    RPA: ({mm/2:.1f}, ~0, ~0) mm")
    
    # ==========================================================================
    # 9. SAVE JSON
    # ==========================================================================
    print(f"\n--- 8. Saving Results ---")
    
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
            "ear_to_ear_mm": mm,
            "scale_factor": float(transform["scale"]),
        },
        "landmarks": {},
        "electrodes": {}
    }
    
    for eid, p in final_pts.items():
        pos_list = p.tolist()
        
        if eid == "INION_EST":
            output["landmarks"]["INION"] = pos_list
        elif isinstance(eid, int):
            name = {0: "NAS", 1: "LPA", 2: "RPA"}.get(eid, f"E{eid - NUM_LANDMARKS}")
            if eid < NUM_LANDMARKS:
                output["landmarks"][name] = pos_list
            else:
                output["electrodes"][name] = pos_list
    
    output["num_electrodes"] = len(output["electrodes"])
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ {OUTPUT_JSON}")
    
    # ==========================================================================
    # 10. SAVE PLY
    # ==========================================================================
    ply_lines = [
        "ply", "format ascii 1.0", 
        f"element vertex {len(final_pts)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ]
    
    for eid, p in final_pts.items():
        if eid == "INION_EST":
            r, g, b = 255, 165, 0  # Orange
        elif isinstance(eid, int) and eid < NUM_LANDMARKS:
            r, g, b = 255, 0, 0    # Red
        else:
            r, g, b = 0, 100, 255  # Blue
        ply_lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}")
    
    with open(OUTPUT_PLY, "w") as f:
        f.write("\n".join(ply_lines))
    print(f"  ✓ {OUTPUT_PLY}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n  Landmarks: {len(output['landmarks'])}")
    print(f"  Electrodes: {output['num_electrodes']}")
    print(f"  Measurement: {method} ({mm:.1f} mm)")


if __name__ == "__main__":
    main()