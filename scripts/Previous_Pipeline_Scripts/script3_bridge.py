"""
================================================================================
SCRIPT 3 v5: SIMPLIFIED 3D PROCESSING
================================================================================

Works with Script 1 v21 (projection-based tracking).

Since Script 1 now already produces 3D positions from multi-view triangulation,
this script focuses on:
    1. Loading the 3D positions from Script 1
    2. Refining with any additional VGGT data
    3. Building head coordinate system
    4. Converting to mm
    5. Saving outputs

================================================================================
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

print("=" * 70)
print("SCRIPT 3 v5: 3D COORDINATE PROCESSING")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRACKING_FILE = os.path.join(RESULTS_DIR, "tracking_results.pkl")
RECON_FILE = os.path.join(RESULTS_DIR, "vggt_output", "reconstruction.npz")
CROP_FILE = os.path.join(RESULTS_DIR, "crop_info.json")
POINTS_3D_FILE = os.path.join(RESULTS_DIR, "points_3d_intermediate.pkl")

OUTPUT_JSON = os.path.join(RESULTS_DIR, "electrodes_3d.json")
OUTPUT_PLY = os.path.join(RESULTS_DIR, "electrodes_3d.ply")
OUTPUT_ELC = os.path.join(RESULTS_DIR, "electrodes.elc")

LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
NUM_LANDMARKS = 3
ELECTRODE_START_ID = 100

LANDMARK_NAMES = {0: "NAS", 1: "LPA", 2: "RPA"}

# Conversion factors
ARC_TO_CHORD = 0.92
CIRCUMFERENCE_TO_EAR = 0.26

print("=" * 70)


# ==============================================================================
# COORDINATE TRANSFORMATION
# ==============================================================================

def script1_to_vggt_coords(u, v, crop_w, crop_h, vggt_size=518):
    scale = vggt_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    pad_w = (vggt_size - new_w) // 2
    pad_h = (vggt_size - new_h) // 2
    return u * scale + pad_w, v * scale + pad_h


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


# ==============================================================================
# HEAD COORDINATE SYSTEM
# ==============================================================================

def estimate_inion_3d(nas, lpa, rpa):
    """Estimate INION from NAS, LPA, RPA."""
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
    
    # INION is opposite to NAS, typically a bit further back
    inion = origin - forward * forward_len * 1.1  # Slightly further than NAS
    
    return inion


def build_head_transform(nas, lpa, rpa, measured_mm):
    """Build transformation to head-centered mm coordinates."""
    inion = estimate_inion_3d(nas, lpa, rpa)
    if inion is None:
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
    centered = point - transform["origin"]
    rotated = transform["rotation"] @ centered
    scaled = rotated * transform["scale"]
    return scaled


# ==============================================================================
# GEOMETRY VALIDATION
# ==============================================================================

def validate_landmark_geometry(nas, lpa, rpa, verbose=True):
    """Check if landmarks make geometric sense."""
    issues = []
    
    ear_vec = rpa - lpa
    ear_dist = np.linalg.norm(ear_vec)
    
    if ear_dist < 0.01:
        issues.append("LPA and RPA are too close together")
        return False, issues
    
    origin = (lpa + rpa) / 2.0
    nas_vec = nas - origin
    nas_dist = np.linalg.norm(nas_vec)
    
    if nas_dist < 0.01:
        issues.append("NAS is too close to ear midpoint")
    
    # Check NAS is roughly perpendicular to ear axis
    ear_axis = ear_vec / ear_dist
    nas_along_ear = abs(np.dot(nas_vec, ear_axis))
    
    if nas_along_ear > 0.5 * nas_dist:
        issues.append(f"NAS is not perpendicular to ear axis")
    
    # Check symmetry
    nas_lpa_dist = np.linalg.norm(nas - lpa)
    nas_rpa_dist = np.linalg.norm(nas - rpa)
    
    asymmetry = abs(nas_lpa_dist - nas_rpa_dist) / max(nas_lpa_dist, nas_rpa_dist)
    if asymmetry > 0.3:
        issues.append(f"NAS-LPA and NAS-RPA distances are asymmetric ({asymmetry*100:.0f}%)")
    
    if verbose and issues:
        print("\n  Geometry warnings:")
        for issue in issues:
            print(f"    - {issue}")
    
    return len(issues) == 0, issues


# ==============================================================================
# MEASUREMENT INPUT
# ==============================================================================

def get_measurement():
    print("\n" + "=" * 60)
    print("HEAD MEASUREMENT")
    print("=" * 60)
    print("[1] Caliper (direct ear-to-ear)")
    print("[2] Tape arc (over top of head)")
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
        print(f"  -> Ear-to-ear (chord): {chord:.1f} mm")
        return chord, "arc"
    elif choice == "3":
        circ = float(input("Enter circumference (mm): "))
        chord = circ * CIRCUMFERENCE_TO_EAR
        print(f"  -> Ear-to-ear: {chord:.1f} mm")
        return chord, "circumference"
    else:
        print("  -> Using default: 150 mm")
        return 150.0, "default"


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    print("\n--- Step 1: Loading Data ---")
    
    # Check if we have pre-computed 3D points from Script 1 v21
    has_3d_points = os.path.exists(POINTS_3D_FILE)
    
    if has_3d_points:
        print("  Found pre-computed 3D points from Script 1")
        with open(POINTS_3D_FILE, "rb") as f:
            positions_3d = pickle.load(f)
        print(f"  Loaded {len(positions_3d)} 3D points")
    else:
        print("  No pre-computed 3D points, will reconstruct from tracking")
        
        # Load tracking and VGGT data
        for f, name in [(TRACKING_FILE, "Tracking"), (RECON_FILE, "Reconstruction"), (CROP_FILE, "Crop info")]:
            if not os.path.exists(f):
                print(f"  ERROR: {name} not found: {f}")
                sys.exit(1)
        
        with open(TRACKING_FILE, "rb") as f:
            tracking = pickle.load(f)
        
        with open(CROP_FILE, "r") as f:
            crop = json.load(f)
        
        recon = np.load(RECON_FILE)
        
        # Build frame mapping
        if "frame_mapping_keys" in recon:
            vggt_indices = recon["frame_mapping_keys"]
        else:
            s1_vals = recon["frame_mapping_values"]
            vggt_indices = np.arange(len(s1_vals))
        
        s1_indices = recon["frame_mapping_values"]
        s1_to_vggt = {int(s1): int(vggt) for vggt, s1 in zip(vggt_indices, s1_indices)}
        
        depths = np.squeeze(recon["depth"])
        intrinsics = np.squeeze(recon["intrinsics"])
        extrinsics = np.squeeze(recon["extrinsics"])
        
        # Ensure 4x4 extrinsics
        if extrinsics.shape[-2:] == (3, 4):
            E4 = np.zeros((len(s1_indices), 4, 4))
            for i in range(len(s1_indices)):
                E4[i] = np.eye(4)
                E4[i, :3, :] = extrinsics[i]
            extrinsics = E4
        
        # Reconstruct 3D points from tracking
        print("\n--- Reconstructing 3D points ---")
        
        all_obj_ids = set()
        for frame_data in tracking.values():
            all_obj_ids.update(frame_data.keys())
        
        positions_3d = {}
        
        for obj_id in tqdm(all_obj_ids, desc="  Processing"):
            observations = []
            
            for s1_idx, frame_tracks in tracking.items():
                if obj_id not in frame_tracks:
                    continue
                if s1_idx not in s1_to_vggt:
                    continue
                
                vggt_idx = s1_to_vggt[s1_idx]
                u, v = frame_tracks[obj_id]
                
                u_vggt, v_vggt = script1_to_vggt_coords(u, v, crop["w"], crop["h"])
                
                p3d = unproject_pixel(
                    u_vggt, v_vggt,
                    depths[vggt_idx],
                    intrinsics[vggt_idx],
                    extrinsics[vggt_idx]
                )
                
                if p3d is not None:
                    observations.append(p3d)
            
            if observations:
                pts = np.array(observations)
                
                # Outlier removal
                if len(pts) >= 3:
                    median = np.median(pts, axis=0)
                    dists = np.linalg.norm(pts - median, axis=1)
                    threshold = np.mean(dists) + 2 * np.std(dists) if np.std(dists) > 0 else np.inf
                    mask = dists < threshold
                    if np.sum(mask) >= 1:
                        positions_3d[obj_id] = np.mean(pts[mask], axis=0)
                    else:
                        positions_3d[obj_id] = median
                else:
                    positions_3d[obj_id] = np.mean(pts, axis=0)
        
        print(f"  Reconstructed {len(positions_3d)} 3D points")
    
    # ==========================================================================
    # STEP 2: VERIFY LANDMARKS
    # ==========================================================================
    print("\n--- Step 2: Verifying Landmarks ---")
    
    missing = []
    for lid, name in [(LANDMARK_NAS, "NAS"), (LANDMARK_LPA, "LPA"), (LANDMARK_RPA, "RPA")]:
        if lid not in positions_3d:
            missing.append(name)
        else:
            pos = positions_3d[lid]
            print(f"  {name}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    if missing:
        print(f"\n  ERROR: Missing landmarks: {', '.join(missing)}")
        sys.exit(1)
    
    nas = positions_3d[LANDMARK_NAS]
    lpa = positions_3d[LANDMARK_LPA]
    rpa = positions_3d[LANDMARK_RPA]
    
    # Validate geometry
    is_valid, issues = validate_landmark_geometry(nas, lpa, rpa)
    
    ear_dist = np.linalg.norm(rpa - lpa)
    print(f"\n  LPA-RPA distance: {ear_dist:.4f} units")
    
    # ==========================================================================
    # STEP 3: GET MEASUREMENT
    # ==========================================================================
    measured_mm, method = get_measurement()
    
    # ==========================================================================
    # STEP 4: BUILD HEAD TRANSFORM
    # ==========================================================================
    print("\n--- Step 4: Building Head Coordinate System ---")
    
    transform = build_head_transform(nas, lpa, rpa, measured_mm)
    
    if transform is None:
        print("  ERROR: Could not build head transform")
        sys.exit(1)
    
    print(f"  Scale: {transform['scale']:.2f} mm/unit")
    
    # ==========================================================================
    # STEP 5: TRANSFORM TO HEAD COORDINATES
    # ==========================================================================
    print("\n--- Step 5: Transforming to mm ---")
    
    final_points = {}
    
    for obj_id, point in positions_3d.items():
        final_points[obj_id] = apply_head_transform(point, transform)
    
    final_points["INION"] = apply_head_transform(transform["inion"], transform)
    
    # ==========================================================================
    # STEP 6: FINAL VERIFICATION
    # ==========================================================================
    print("\n--- Step 6: Final Verification ---")
    
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
    
    # Sanity checks
    print(f"\n  Sanity checks:")
    
    if lpa_f[0] < 0 and rpa_f[0] > 0:
        print(f"    [OK] LPA is left (X={lpa_f[0]:.1f}), RPA is right (X={rpa_f[0]:.1f})")
    else:
        print(f"    [!] LPA/RPA may be swapped!")
    
    if nas_f[1] > 0 and inion_f[1] < 0:
        print(f"    [OK] NAS is front (Y={nas_f[1]:.1f}), INION is back (Y={inion_f[1]:.1f})")
    else:
        print(f"    [!] NAS/INION orientation may be wrong!")
    
    lpa_rpa_y_diff = abs(lpa_f[1] - rpa_f[1])
    lpa_rpa_z_diff = abs(lpa_f[2] - rpa_f[2])
    if lpa_rpa_y_diff < 20 and lpa_rpa_z_diff < 20:
        print(f"    [OK] LPA-RPA symmetric (Y diff: {lpa_rpa_y_diff:.1f}, Z diff: {lpa_rpa_z_diff:.1f})")
    else:
        print(f"    [!] LPA-RPA not symmetric (Y diff: {lpa_rpa_y_diff:.1f}, Z diff: {lpa_rpa_z_diff:.1f})")
    
    # ==========================================================================
    # STEP 7: SAVE JSON
    # ==========================================================================
    print("\n--- Step 7: Saving Results ---")
    
    output = {
        "coordinate_system": {
            "origin": "midpoint between LPA and RPA",
            "x_axis": "left to right (LPA -> RPA)",
            "y_axis": "back to front (INION -> NAS)",
            "z_axis": "down to up",
        },
        "units": "mm",
        "measurement": {
            "method": method,
            "ear_to_ear_mm": measured_mm,
            "scale_factor": float(transform["scale"]),
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
        elif isinstance(obj_id, int) and obj_id >= ELECTRODE_START_ID:
            output["electrodes"][f"E{obj_id - ELECTRODE_START_ID}"] = pos_list
    
    output["num_electrodes"] = len(output["electrodes"])
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"[OK] {OUTPUT_JSON}")
    
    # ==========================================================================
    # STEP 8: SAVE PLY
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
            r, g, b = 255, 165, 0
        elif obj_id == LANDMARK_NAS:
            r, g, b = 255, 0, 0
        elif obj_id == LANDMARK_LPA:
            r, g, b = 0, 0, 255
        elif obj_id == LANDMARK_RPA:
            r, g, b = 0, 255, 0
        else:
            r, g, b = 0, 100, 255
        
        ply_lines.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}")
    
    with open(OUTPUT_PLY, "w") as f:
        f.write("\n".join(ply_lines))
    
    print(f"[OK] {OUTPUT_PLY}")
    
    # ==========================================================================
    # STEP 9: SAVE ELC (for EEG software compatibility)
    # ==========================================================================
    elc_lines = [
        "# Electrode positions",
        "# Generated by EEG Electrode Registration Pipeline",
        f"NumberPositions= {len(output['electrodes']) + 3}",
        "UnitPosition	mm",
        "Positions"
    ]
    
    # Add fiducials first
    for name in ["NAS", "LPA", "RPA"]:
        pos = output["landmarks"][name]
        elc_lines.append(f"{name}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    
    # Add electrodes
    for ename, pos in sorted(output["electrodes"].items()):
        elc_lines.append(f"{ename}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    
    elc_lines.append("Labels")
    elc_lines.append("NAS\tLPA\tRPA\t" + "\t".join(sorted(output["electrodes"].keys())))
    
    with open(OUTPUT_ELC, "w") as f:
        f.write("\n".join(elc_lines))
    
    print(f"[OK] {OUTPUT_ELC}")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SCRIPT 3 COMPLETE!")
    print("=" * 70)
    print(f"\n  Landmarks: {len(output['landmarks'])}")
    print(f"  Electrodes: {output['num_electrodes']}")
    print(f"\n  Output files:")
    print(f"    - {OUTPUT_JSON}")
    print(f"    - {OUTPUT_PLY}")
    print(f"    - {OUTPUT_ELC}")


if __name__ == "__main__":
    main()