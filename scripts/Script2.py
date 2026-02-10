"""
SCRIPT 2: 3D COORDINATE PROCESSING

Converts VGGT arbitrary units to real-world mm coordinates.

Since Script 1 already produces 3D positions from multi-view triangulation,
this script focuses on:
    1. Loading the 3D positions from Script 1
    2. Building head coordinate system from landmarks (NAS, LPA, RPA)
    3. Converting to mm using ear-to-ear measurement
    4. Saving outputs (.json, .ply, .elc)
"""

import os
import sys
import json
import pickle
import numpy as np


print("SCRIPT 2: 3D COORDINATE PROCESSING")


# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results")

LANDMARK_NAS = 0
LANDMARK_LPA = 1
LANDMARK_RPA = 2
ELECTRODE_START_ID = 100

LANDMARK_NAMES = {0: "NAS", 1: "LPA", 2: "RPA"}

# Conversion factors for different measurement methods
ARC_TO_CHORD = 0.92
CIRCUMFERENCE_TO_EAR = 0.26


def select_results_directory():
    """
    Let user select which video's results to process.
    
    Returns:
        dict with all file paths for the selected video
    """
    if not os.path.exists(RESULTS_BASE_DIR):
        print(f"ERROR: Results directory not found: {RESULTS_BASE_DIR}")
        print("Please run script1.py first.")
        sys.exit(1)
    
    # Get all video result directories
    video_dirs = []
    for item in os.listdir(RESULTS_BASE_DIR):
        item_path = os.path.join(RESULTS_BASE_DIR, item)
        if os.path.isdir(item_path):
            # Check if it has required files
            points_3d_file = os.path.join(item_path, "points_3d_intermediate.pkl")
            if os.path.exists(points_3d_file):
                video_dirs.append(item)
    
    if not video_dirs:
        print("ERROR: No completed video results found in results/")
        print("Please run script1.py first.")
        sys.exit(1)
    
    video_dirs.sort()
    
    print("\nSELECT VIDEO RESULTS TO PROCESS")
    print("\nAvailable results:")
    for i, video_name in enumerate(video_dirs, 1):
        print(f"  [{i}] {video_name}")
    print()
    
    while True:
        try:
            choice = input(f"Enter (1-{len(video_dirs)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(video_dirs):
                break
            print(f"Please enter a number between 1 and {len(video_dirs)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_video = video_dirs[idx]
    results_dir = os.path.join(RESULTS_BASE_DIR, selected_video)
    
    paths = {
        "video_name": selected_video,
        "results_dir": results_dir,
        "points_3d_file": os.path.join(results_dir, "points_3d_intermediate.pkl"),
        "output_json": os.path.join(results_dir, "electrodes_3d.json"),
        "output_ply": os.path.join(results_dir, "electrodes_3d.ply"),
        "output_elc": os.path.join(results_dir, "electrodes.elc"),
    }

    print(f"\nPROCESSING: {selected_video}")
    print(f"Results directory: {results_dir}")
    
    return paths


# HEAD COORDINATE SYSTEM


def estimate_inion_3d(nas, lpa, rpa):
    """Estimate INION from NAS, LPA, RPA using anatomical proportions."""
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
    
    up = np.cross(ear_axis, forward)
    up = up / np.linalg.norm(up)
    
    inion_distance = forward_len * 1.05
    z_offset = -0.08 * ear_len
    inion = origin - forward * inion_distance + up * z_offset
    
    return inion


def build_head_transform(nas, lpa, rpa, measured_mm):
    """Build transformation to head-centered mm coordinates."""
    inion = estimate_inion_3d(nas, lpa, rpa)
    if inion is None:
        return None
    
    origin = (lpa + rpa) / 2.0
    
    # X-axis: left to right (LPA -> RPA)
    x_axis = rpa - lpa
    raw_ear_dist = np.linalg.norm(x_axis)
    x_axis = x_axis / raw_ear_dist
    
    # Y-axis: back to front (INION -> NAS)
    y_vec = nas - inion
    y_axis = y_vec - np.dot(y_vec, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Z-axis: down to up
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
    """Transform a point to head-centered mm coordinates."""
    centered = point - transform["origin"]
    rotated = transform["rotation"] @ centered
    scaled = rotated * transform["scale"]
    return scaled


# MEASUREMENT INPUT


def get_measurement():
    """Get ear-to-ear measurement from user."""
    print("\n=== HEAD MEASUREMENT ===")
    print("Select measurement method:")
    print("  [1] Caliper (direct ear-to-ear)")
    print("  [2] Tape arc (over top of head)")
    print("  [3] Head circumference")
    print("  [4] Default (150mm)")
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


# MAIN PIPELINE


def main():
    # SELECT WHICH VIDEO TO PROCESS
    paths = select_results_directory()
    
    POINTS_3D_FILE = paths["points_3d_file"]
    OUTPUT_JSON = paths["output_json"]
    OUTPUT_PLY = paths["output_ply"]
    OUTPUT_ELC = paths["output_elc"]
    
    # STEP 1: LOAD 3D POINTS
    print("\n=== Step 1: Loading 3D Points ===")
    
    if not os.path.exists(POINTS_3D_FILE):
        print(f"ERROR: 3D points file not found: {POINTS_3D_FILE}")
        print("Please run script1.py first.")
        sys.exit(1)
    
    with open(POINTS_3D_FILE, "rb") as f:
        positions_3d = pickle.load(f)
    print(f"  Loaded {len(positions_3d)} 3D points")
    
    # STEP 2: VERIFY LANDMARKS
    print("\n=== Step 2: Verifying Landmarks ===")
    
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
    
    ear_dist = np.linalg.norm(rpa - lpa)
    print(f"\n  LPA-RPA distance: {ear_dist:.4f} units")
    
    # STEP 3: GET MEASUREMENT
    measured_mm, method = get_measurement()
    
    # STEP 4: BUILD HEAD TRANSFORM
    print("\n=== Step 4: Building Head Coordinate System ===")
    
    transform = build_head_transform(nas, lpa, rpa, measured_mm)
    if transform is None:
        print("  ERROR: Could not build head transform")
        sys.exit(1)
    
    print(f"  Scale: {transform['scale']:.2f} mm/unit")
    
    # STEP 5: TRANSFORM ALL POINTS TO MM
    print("\n=== Step 5: Transforming to mm ===")
    
    final_points = {}
    for obj_id, point in positions_3d.items():
        final_points[obj_id] = apply_head_transform(point, transform)
    
    # Add estimated INION
    final_points["INION"] = apply_head_transform(transform["inion"], transform)
    
    # Print landmark positions
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
    print(f"    NAS-INION:  {final_nasi:.1f} mm")
    
    # STEP 6: SAVE JSON
    print("\n=== Step 6: Saving Results ===")
    
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
        "electrodes": {},
    }
    
    for obj_id, pos in final_points.items():
        pos_list = pos.tolist()
        
        if obj_id == "INION":
            output["landmarks"]["INION"] = {"position": pos_list}
        elif obj_id == LANDMARK_NAS:
            output["landmarks"]["NAS"] = {"position": pos_list}
        elif obj_id == LANDMARK_LPA:
            output["landmarks"]["LPA"] = {"position": pos_list}
        elif obj_id == LANDMARK_RPA:
            output["landmarks"]["RPA"] = {"position": pos_list}
        elif isinstance(obj_id, int) and obj_id >= ELECTRODE_START_ID:
            output["electrodes"][f"E{obj_id - ELECTRODE_START_ID}"] = {"position": pos_list}
    
    output["num_electrodes"] = len(output["electrodes"])
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [OK] {OUTPUT_JSON}")
    
    # STEP 7: SAVE PLY
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
            r, g, b = 255, 165, 0  # Orange
        elif obj_id == LANDMARK_NAS:
            r, g, b = 255, 0, 0    # Red
        elif obj_id == LANDMARK_LPA:
            r, g, b = 0, 0, 255    # Blue
        elif obj_id == LANDMARK_RPA:
            r, g, b = 0, 255, 0    # Green
        else:
            r, g, b = 0, 100, 255  # Light blue for electrodes
        ply_lines.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r} {g} {b}")
    
    with open(OUTPUT_PLY, "w") as f:
        f.write("\n".join(ply_lines))
    print(f"  [OK] {OUTPUT_PLY}")
    
    # STEP 8: SAVE ELC
    elc_lines = [
        "# Electrode positions",
        "# Generated by EEG Electrode Registration Pipeline",
        f"NumberPositions= {len(output['electrodes']) + 3}",
        "UnitPosition	mm",
        "Positions"
    ]
    
    for name in ["NAS", "LPA", "RPA"]:
        pos = output["landmarks"][name]["position"]
        elc_lines.append(f"{name}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    
    for ename in sorted(output["electrodes"].keys(), key=lambda x: int(x[1:])):
        pos = output["electrodes"][ename]["position"]
        elc_lines.append(f"{ename}\t{pos[0]:.2f}\t{pos[1]:.2f}\t{pos[2]:.2f}")
    
    elc_lines.append("Labels")
    elc_lines.append("NAS\tLPA\tRPA\t" + "\t".join(sorted(output["electrodes"].keys(), key=lambda x: int(x[1:]))))
    
    with open(OUTPUT_ELC, "w") as f:
        f.write("\n".join(elc_lines))
    print(f"  [OK] {OUTPUT_ELC}")
    
    # SUMMARY
    print("SCRIPT 2 COMPLETE!")
    print(f"\n  Landmarks: {len(output['landmarks'])}")
    print(f"  Electrodes: {output['num_electrodes']}")
    print(f"\n  Output files:")
    print(f"    - {OUTPUT_JSON}")
    print(f"    - {OUTPUT_PLY}")
    print(f"    - {OUTPUT_ELC}")


if __name__ == "__main__":
    main()