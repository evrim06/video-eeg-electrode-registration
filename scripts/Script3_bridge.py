"""
VGGT-YOLO Bridge Script (Step 3)

1. Matches 2D Tracking (Script 1) to 3D Depth (Script 2).
2. Calculates 3D Inion from NAS/LPA/RPA.
3. Aligns Head to Origin with proper scaling.
"""

import os
import sys
import json
import pickle
import numpy as np

# CONFIG
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

# Standard head dimensions for scaling
STANDARD_EAR_TO_EAR_MM = 150.0


# MATH HELPERS


def transform_coords_to_vggt_space(u, v, crop_w, crop_h, vggt_size=518):
    """Transform coordinates from CROPPED frame space to VGGT 518x518 space."""
    scale = vggt_size / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    pad_w, pad_h = (vggt_size - new_w) // 2, (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


def unproject_point(u, v, depth_map, intrinsic, extrinsic):
    """Unproject 2D pixel to 3D world coordinates."""
    H, W = depth_map.shape
    
    # Bounds check
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return None
    
    # Bilinear interpolation for sub-pixel accuracy
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
    P_world = np.linalg.inv(extrinsic) @ P_cam
    
    return P_world[:3]


def estimate_inion_3d(nas, lpa, rpa):
    """
    Estimate INION position in 3D.
    INION lies opposite to NAS, perpendicular to the LPA-RPA axis.
    """
    nas = np.array(nas)
    lpa = np.array(lpa)
    rpa = np.array(rpa)
    
    # Origin: midpoint between ears
    origin = (lpa + rpa) / 2.0
    
    # Ear axis (X direction)
    ear_axis = rpa - lpa
    ear_axis = ear_axis / np.linalg.norm(ear_axis)
    
    # Vector from origin to NAS
    nas_vec = nas - origin
    
    # Remove component parallel to ear axis (orthogonalize)
    forward_dir = nas_vec - np.dot(nas_vec, ear_axis) * ear_axis
    forward_len = np.linalg.norm(forward_dir)
    
    if forward_len < 1e-6:
        return None
    
    forward_dir = forward_dir / forward_len
    
    # INION is in the opposite direction
    inion_pos = origin - forward_dir * forward_len
    
    return inion_pos


def define_head_coordinate_system(landmarks_3d):
    """
    Define anatomical head coordinate system.
    
    Returns:
        - origin: midpoint between ears
        - rotation: 3x3 rotation matrix [X, Y, Z]
        - scale: factor to convert to mm (ear-to-ear = 150mm)
        - estimated_inion: 3D position
    """
    NAS = np.array(landmarks_3d[LANDMARK_NAS])
    LPA = np.array(landmarks_3d[LANDMARK_LPA])
    RPA = np.array(landmarks_3d[LANDMARK_RPA])
    
    # Estimate INION
    INION = estimate_inion_3d(NAS, LPA, RPA)
    if INION is None:
        print("  Could not estimate INION")
        return None
    
    # Origin: midpoint between ears
    origin = (LPA + RPA) / 2.0
    
    # X-axis: Left to Right
    x_axis = RPA - LPA
    ear_to_ear_dist = np.linalg.norm(x_axis)
    x_axis = x_axis / ear_to_ear_dist
    
    # Y-axis: Back to Front (INION -> NAS)
    y_dir = NAS - INION
    # Orthogonalize to X
    y_axis = y_dir - np.dot(y_dir, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Z-axis: Down to Up (cross product)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Rotation matrix
    R = np.array([x_axis, y_axis, z_axis])
    
    # Scale: normalize ear-to-ear to standard 150mm
    scale = STANDARD_EAR_TO_EAR_MM / ear_to_ear_dist
    
    return {
        "origin": origin,
        "rotation": R,
        "scale": scale,
        "ear_to_ear_raw": ear_to_ear_dist,
        "estimated_inion": INION
    }


def transform_to_head_coords(point, transform):
    """Transform a 3D point to head-centered coordinates (in mm)."""
    p = np.array(point)
    origin = transform["origin"]
    R = transform["rotation"]
    scale = transform["scale"]
    
    p_centered = p - origin
    p_rotated = R @ p_centered
    p_scaled = p_rotated * scale
    
    return p_scaled



# MAIN


def main():
    print("=" * 70)
    print("BRIDGE SCRIPT (STEP 3)")
    print("=" * 70)


    # 1. LOAD DATA

    print("\n--- 1. Loading Data ---")
    
    if not os.path.exists(TRACKING_FILE):
        print(f"Missing: {TRACKING_FILE}")
        print("   Run Script 1 first!")
        sys.exit(1)
        
    if not os.path.exists(RECON_FILE):
        print(f"Missing: {RECON_FILE}")
        print("   Run Script 2 first!")
        sys.exit(1)

    with open(TRACKING_FILE, "rb") as f:
        tracking_data = pickle.load(f)
    
    with open(CROP_INFO_FILE, "r") as f:
        crop = json.load(f)
    
    recon = np.load(RECON_FILE)
    
    # Load frame mapping
    if "frame_mapping_keys" not in recon:
        print("Reconstruction file missing frame mapping.")
        print("   Re-run Script 2!")
        sys.exit(1)
    
    # Build mapping: Script 1 frame index -> VGGT frame index
    vggt_indices = recon["frame_mapping_keys"]
    script1_indices = recon["frame_mapping_values"]
    s1_to_vggt = {int(s1): int(v) for v, s1 in zip(vggt_indices, script1_indices)}
    
    print(f"  ✓ Tracking data: {len(tracking_data)} frames")
    print(f"  ✓ VGGT frames: {len(s1_to_vggt)} with depth maps")
    print(f"  ✓ Crop: {crop['w']}x{crop['h']} at ({crop['x']}, {crop['y']})")


    # 2. UNPROJECT 2D -> 3D

    print("\n--- 2. Unprojecting 2D to 3D ---")
    
    # Collect all 3D observations per electrode
    points_3d_all = {}  # {electrode_id: [list of 3D points]}
    
    frames_matched = 0
    points_unprojected = 0
    
    for s1_idx, tracks in tracking_data.items():
        # Check if this frame has VGGT depth
        if s1_idx not in s1_to_vggt:
            continue
        
        vggt_idx = s1_to_vggt[s1_idx]
        frames_matched += 1
        
        # Get depth and camera data for this frame
        depth = recon["depth"][vggt_idx]
        intrinsic = recon["intrinsics"][vggt_idx]
        extrinsic = recon["extrinsics"][vggt_idx]
        
        for eid, (u, v) in tracks.items():
            # IMPORTANT: tracking coords are in CROPPED space
            # (If Script 1 was fixed to not add offset)
            # If Script 1 still adds offset, uncomment these lines:
            # u = u - crop["x"]
            # v = v - crop["y"]
            
            # Transform to VGGT 518x518 space
            u_vggt, v_vggt = transform_coords_to_vggt_space(
                u, v, crop["w"], crop["h"]
            )
            
            # Unproject to 3D
            p3d = unproject_point(u_vggt, v_vggt, depth, intrinsic, extrinsic)
            
            if p3d is not None:
                if eid not in points_3d_all:
                    points_3d_all[eid] = []
                points_3d_all[eid].append(p3d)
                points_unprojected += 1
    
    print(f" Frames matched: {frames_matched}")
    print(f" Points unprojected: {points_unprojected}")
    print(f" Unique electrodes: {len(points_3d_all)}")


    # 3. ROBUST AVERAGING

    print("\n--- 3. Robust Averaging ---")
    
    # Average each electrode's observations with outlier removal
    points_3d_avg = {}
    
    for eid, pts_list in points_3d_all.items():
        pts = np.array(pts_list)
        
        # Outlier removal (2σ threshold)
        if len(pts) > 3:
            mean = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            threshold = np.mean(dists) + 2.0 * np.std(dists)
            mask = dists < threshold
            if np.sum(mask) >= 3:
                pts = pts[mask]
        
        # Final average
        points_3d_avg[eid] = np.mean(pts, axis=0)
        
        # Label for printing
        if eid < NUM_LANDMARKS:
            label = ["NAS", "LPA", "RPA"][eid]
        else:
            label = f"E{eid - NUM_LANDMARKS}"
        
        print(f"    {label}: {len(pts_list)} obs -> ({points_3d_avg[eid][0]:.3f}, {points_3d_avg[eid][1]:.3f}, {points_3d_avg[eid][2]:.3f})")

    # 4. HEAD ALIGNMENT

    print("\n--- 4. Head Coordinate Alignment ---")
    
    # Check we have all landmarks
    if not all(lid in points_3d_avg for lid in [LANDMARK_NAS, LANDMARK_LPA, LANDMARK_RPA]):
        print(" Missing landmarks! Cannot align.")
        sys.exit(1)
    
    # Define coordinate system
    transform = define_head_coordinate_system(points_3d_avg)
    
    if transform is None:
        print(" Failed to define coordinate system!")
        sys.exit(1)
    
    print(f"  ✓ Ear-to-ear distance (raw): {transform['ear_to_ear_raw']:.4f}")
    print(f"  ✓ Scale factor: {transform['scale']:.4f}")
    print(f"  ✓ Estimated INION: ({transform['estimated_inion'][0]:.3f}, {transform['estimated_inion'][1]:.3f}, {transform['estimated_inion'][2]:.3f})")
    
    # Transform all points to head coordinates
    aligned_points = {}
    
    for eid, p in points_3d_avg.items():
        aligned_points[eid] = transform_to_head_coords(p, transform)
    
    # Add estimated INION (use a special key, NOT 3!)
    aligned_points["INION_EST"] = transform_to_head_coords(transform["estimated_inion"], transform)


    # 5. SAVE JSON

    print("\n--- 5. Saving Results ---")
    
    output = {
        "coordinate_system": {
            "type": "head_aligned_3d",
            "origin": "midpoint between LPA and RPA",
            "x_axis": "LPA to RPA (left to right)",
            "y_axis": "INION to NAS (back to front)",
            "z_axis": "inferior to superior (down to up)",
        },
        "units": "mm",
        "scaling": {
            "method": "ear_to_ear_normalization",
            "target_ear_to_ear_mm": STANDARD_EAR_TO_EAR_MM,
            "raw_ear_to_ear": float(transform["ear_to_ear_raw"]),
            "scale_factor": float(transform["scale"]),
        },
        "landmarks": {},
        "electrodes": {},
    }
    
    for eid, pos in aligned_points.items():
        pos_list = pos.tolist()
        
        # Handle special INION key
        if eid == "INION_EST":
            output["landmarks"]["INION"] = pos_list
            output["landmarks"]["INION_note"] = "Estimated from NAS, LPA, RPA geometry"
        elif eid == LANDMARK_NAS:
            output["landmarks"]["NAS"] = pos_list
        elif eid == LANDMARK_LPA:
            output["landmarks"]["LPA"] = pos_list
        elif eid == LANDMARK_RPA:
            output["landmarks"]["RPA"] = pos_list
        else:
            # Electrodes: ID 3 -> E0, ID 4 -> E1, etc.
            output["electrodes"][f"E{eid - NUM_LANDMARKS}"] = pos_list
    
    output["num_electrodes"] = len(output["electrodes"])
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f" Saved: {OUTPUT_JSON}")
    

    # 6. SAVE PLY
    with open(OUTPUT_PLY, "w") as f:
        # Count vertices (exclude special keys)
        num_verts = sum(1 for k in aligned_points if isinstance(k, int) or k == "INION_EST")
        
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        
        for eid, pos in aligned_points.items():
            if eid == "INION_EST":
                r, g, b = 255, 165, 0  # Orange for estimated INION
            elif isinstance(eid, int) and eid < NUM_LANDMARKS:
                r, g, b = 255, 0, 0    # Red for landmarks
            else:
                r, g, b = 0, 100, 255  # Blue for electrodes
            
            f.write(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}\n")
    
    print(f" Saved: {OUTPUT_PLY}")
    

    # SUMMARY
 
    print("\n" + "=" * 70)
    print("BRIDGE COMPLETE!")
    print("=" * 70)
    print(f"\n  Landmarks: {len(output['landmarks'])}")
    print(f"  Electrodes: {output['num_electrodes']}")
    print(f"  Units: {output['units']}")
    print(f"\n  Output files:")
    print(f"    - {OUTPUT_JSON}")
    print(f"    - {OUTPUT_PLY}")


if __name__ == "__main__":
    main()