"""
VGGT-YOLO Bridge Script (Step 3) 

Outputs REAL 3D electrode positions in millimeters.
Features:
1. Matches 2D Tracking to 3D Depth (using explicit frame mapping).
2. Estimates 3D Inion from NAS/LPA/RPA geometry.
3. SCALES the model using real-world user measurements.
"""

import os
import sys
import json
import pickle
import numpy as np

# CONFIGURATION


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
CROP_INFO_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "electrodes_3d.json")
OUTPUT_PLY = os.path.join(RESULTS_DIR, "electrodes_3d.ply")

# Landmarks
LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3

# Conversion factors
ARC_TO_CHORD_RATIO = 0.92  # chord ≈ 92% of arc (tape measure vs caliper)
CIRCUMFERENCE_TO_EAR_RATIO = 0.26  # ear-to-ear ≈ 26% of circumference


# MEASUREMENT INPUT


def get_ear_to_ear_measurement():
    """Get ear-to-ear chord distance from user via multiple options."""
    print("\n" + "=" * 70)
    print("HEAD MEASUREMENT REQUIRED")
    print("=" * 70)
    print("\nTo output real millimeters, we need to scale the 3D model.")
    print("Choose a measurement method:\n")
    print("  [1] Direct ear-to-ear (Caliper straight line)")
    print("  [2] Tape measure over head (Arc: Left Ear -> Top -> Right Ear)")
    print("  [3] Head circumference (Tape around head)")
    print("  [4] Skip - use average adult value (150mm)\n")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("\nMeasure straight-line distance between Tragus points.")
            val = float(input("Enter distance in mm: "))
            return val, "direct_caliper"
        
        elif choice == "2":
            print("\nMeasure arc from Left Tragus -> Vertex -> Right Tragus.")
            arc = float(input("Enter arc length in mm: "))
            chord = arc * ARC_TO_CHORD_RATIO
            print(f"  -> Estimated Chord: {chord:.1f} mm")
            return chord, "arc_converted"
            
        elif choice == "3":
            print("\nMeasure circumference above eyebrows and ears.")
            circ = float(input("Enter circumference in mm: "))
            chord = circ * CIRCUMFERENCE_TO_EAR_RATIO
            print(f"  -> Estimated Chord: {chord:.1f} mm")
            return chord, "circumference_converted"
            
        elif choice == "4":
            print("\n Using average adult value: 150mm")
            return 150.0, "average_default"
            
        else:
            print("Invalid choice.")

# MATH HELPERS


def transform_coords_to_vggt_space(u, v, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    pad_w, pad_h = (vggt_size - new_w) // 2, (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h

def unproject_point(u, v, depth_map, intrinsic, extrinsic):
    H, W = depth_map.shape
    if not (0 <= u < W - 1 and 0 <= v < H - 1): return None
    
    # Bilinear interpolation for smoother depth
    u0, v0 = int(u), int(v)
    u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
    du, dv = u - u0, v - v0
    
    z = (depth_map[v0, u0] * (1 - du) * (1 - dv) +
         depth_map[v0, u1] * du * (1 - dv) +
         depth_map[v1, u0] * (1 - du) * dv +
         depth_map[v1, u1] * du * dv)
         
    if z <= 0 or not np.isfinite(z): return None

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    
    P_cam = np.array([x_cam, y_cam, z, 1.0])
    return (np.linalg.inv(extrinsic) @ P_cam)[:3]

def estimate_inion_3d(nas, lpa, rpa):
    """Calculates 3D Inion position based on NAS and Ears."""
    origin = (lpa + rpa) / 2.0
    nas_vec = nas - origin
    
    ear_axis = rpa - lpa
    ear_axis /= np.linalg.norm(ear_axis)
    
    # Project NAS vector backward (perpendicular to ears)
    forward_dir = nas_vec - np.dot(nas_vec, ear_axis) * ear_axis
    forward_len = np.linalg.norm(forward_dir)
    forward_dir /= forward_len
    
    inion_vec = -forward_dir
    inion_pos = origin + (inion_vec * forward_len) # Assume symmetric depth
    return inion_pos

def define_head_coordinate_system(points, measured_ear_to_ear_mm):
    """Calculates transform to align head and scale to real mm."""
    NAS = np.array(points[LANDMARK_NAS])
    LPA = np.array(points[LANDMARK_LPA])
    RPA = np.array(points[LANDMARK_RPA])
    
    INION = estimate_inion_3d(NAS, LPA, RPA)
    
    # Origin: Midpoint of ears
    origin = (LPA + RPA) / 2.0
    
    # X-Axis: Left to Right Ear
    x_axis = RPA - LPA
    raw_dist = np.linalg.norm(x_axis)
    x_axis /= raw_dist
    
    # Y-Axis: Back to Front (Inion -> Nas)
    y_dir = NAS - INION
    y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis # Orthogonalize
    y_axis /= np.linalg.norm(y_axis)
    
    # Z-Axis: Up
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    # Rotation Matrix
    R = np.array([x_axis, y_axis, z_axis])
    
    # Scale Factor (Real mm / Raw VGGT units)
    scale = measured_ear_to_ear_mm / raw_dist
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "estimated_inion": INION
    }

def transform_point(point, transform):
    p = np.array(point)
    # 1. Center
    p -= transform["origin"]
    # 2. Rotate
    p = transform["rotation"] @ p
    # 3. Scale
    p *= transform["scale"]
    return p


# MAIN


def main():
    print("=" * 70)
    print("BRIDGE SCRIPT (STEP 3) - REAL MILLIMETERS")
    print("=" * 70)
    
    # 0. Get Measurement
    ear_to_ear_mm, method = get_ear_to_ear_measurement()
    print(f"\n Using Scale Reference: {ear_to_ear_mm:.1f} mm ({method})")

    # 1. Load Data
    if not os.path.exists(TRACKING_FILE) or not os.path.exists(RECON_FILE):
        print(" Missing input files."); sys.exit(1)

    with open(TRACKING_FILE, "rb") as f: tracking_data = pickle.load(f)
    with open(CROP_INFO_FILE, "r") as f: crop = json.load(f)
    recon = np.load(RECON_FILE)
    
    # Load Explicit Mapping (CRITICAL FIX)
    if "frame_mapping_keys" not in recon:
        print(" Error: Old reconstruction file. Re-run Script 2.")
        sys.exit(1)
    
    s1_to_vggt = {int(s1): int(v) for v, s1 in zip(recon["frame_mapping_keys"], recon["frame_mapping_values"])}
    print(f"\n Loaded Frame Mapping ({len(s1_to_vggt)} frames)")

    # 2. Unproject
    print("\n--- Unprojecting 2D to 3D ---")
    points_3d_all = {}
    
    for s1_idx, tracks in tracking_data.items():
        if s1_idx not in s1_to_vggt: continue
        vggt_idx = s1_to_vggt[s1_idx]
        
        depth = recon["depth"][vggt_idx]
        intr = recon["intrinsics"][vggt_idx]
        extr = recon["extrinsics"][vggt_idx]
        
        for eid, (u, v) in tracks.items():
            u_v, v_v = transform_coords_to_vggt_space(u, v, crop["w"], crop["h"])
            p3d = unproject_point(u_v, v_v, depth, intr, extr)
            if p3d is not None:
                if eid not in points_3d_all: points_3d_all[eid] = []
                points_3d_all[eid].append(p3d)

    # 3. Robust Average
    points_3d_avg = {}
    for eid, pts_list in points_3d_all.items():
        pts = np.array(pts_list)
        if len(pts) > 3:
            mean = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            pts = pts[dists < (np.mean(dists) + 2.0 * np.std(dists))]
        points_3d_avg[eid] = np.mean(pts, axis=0)

    # 4. Align & Scale
    print("\n--- Aligning & Scaling ---")
    if not all(k in points_3d_avg for k in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]):
        print(" Error: Missing landmarks (NAS, LPA, or RPA). Cannot align.")
        sys.exit(1)
        
    transform = define_head_coordinate_system(points_3d_avg, ear_to_ear_mm)
    print(f" Scale Factor applied: {transform['scale']:.2f}")
    
    aligned_points = {}
    for eid, p in points_3d_avg.items():
        aligned_points[eid] = transform_point(p, transform)
        
    # Add Estimated Inion to output
    inion_aligned = transform_point(transform["estimated_inion"], transform)
    aligned_points["INION_EST"] = inion_aligned

    # 5. Save Results
    output = {
        "units": "mm",
        "measurement_method": method,
        "landmarks": {},
        "electrodes": {}
    }
    
    for eid, pos in aligned_points.items():
        pos_list = pos.tolist()
        if eid == "INION_EST":
            output["landmarks"]["INION"] = pos_list
        elif isinstance(eid, int):
            # FIXED LINE BELOW
            name = {0:"NAS", 1:"LPA", 2:"RPA"}.get(eid, f"E{eid-NUM_LANDMARKS}")
            if eid < NUM_LANDMARKS: output["landmarks"][name] = pos_list
            else: output["electrodes"][name] = pos_list

    with open(OUTPUT_JSON, "w") as f: json.dump(output, f, indent=2)
    print(f" Saved JSON: {OUTPUT_JSON}")
    
    # 6. Save PLY
    with open(OUTPUT_PLY, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(aligned_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for eid, pos in aligned_points.items():
            if eid == "INION_EST": r,g,b = 255,165,0 # Orange for Est Inion
            elif isinstance(eid, int) and eid < NUM_LANDMARKS: r,g,b = 255,0,0
            else: r,g,b = 0,100,255
            f.write(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}\n")
    print(f" Saved PLY: {OUTPUT_PLY}")

if __name__ == "__main__":
    main()