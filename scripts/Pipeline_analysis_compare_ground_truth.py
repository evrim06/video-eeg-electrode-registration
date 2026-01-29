import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
from glob import glob
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# Configuration
RESULTS_DIR = "results"
SCANNER_DIR = "Scanner_recordings"
OUTPUT_DIR = "validation_results"
LANDMARKS = ["NAS", "LPA", "RPA", "INION"]

# Path to your standard montage file
STANDARD_MONTAGE_PATH = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\mobile24.elp"

def parse_elp_montage(elp_path):
    """Parse .elp file to get standard 2D electrode positions.
    
    .elp format example:
    # ASA electrode file
    ReferenceLabel  avg
    UnitPosition    mm
    NumberPositions=   24
    Positions
    -29.5   87.9   -32.4
    Labels
    Fp1
    ...
    """
    positions_2d = {}
    
    if not os.path.exists(elp_path):
        print(f"Warning: Standard montage file not found: {elp_path}")
        return None
    
    with open(elp_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Find where positions and labels start
    in_positions = False
    in_labels = False
    position_list = []
    label_list = []
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
            
        if line.lower().startswith('positions'):
            in_positions = True
            in_labels = False
            continue
        
        if line.lower().startswith('labels'):
            in_positions = False
            in_labels = True
            continue
        
        if in_positions:
            # Parse position line: X Y Z (we'll use X and Y for 2D)
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    position_list.append((x, y, z))
                except ValueError:
                    continue
        
        if in_labels:
            # Parse label
            label = line.strip().upper()
            if label and not label.startswith('#'):
                label_list.append(label)
    
    # Match labels with positions
    for i, label in enumerate(label_list):
        if i < len(position_list):
            x, y, z = position_list[i]
            # For head map plotting, we use X and Y coordinates
            # The plot function expects a 3D position but uses [1] and [0] for x,y
            # So we store as [y, x, z] to match the plotting convention
            positions_2d[label] = np.array([y, x, z])  # Note: swapped for plotting
    
    print(f"✓ Loaded {len(positions_2d)} electrode positions from standard montage")
    return positions_2d

def parse_pipeline_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    points = {}
    for source in [data.get("landmarks", {}), data.get("electrodes", {})]:
        for k, v in source.items():
            points[k.upper()] = np.array(v["position"])
    return points

def parse_scanner_elc(path):
    points = {}
    in_positions = False
    if not os.path.exists(path): return points
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.lower().startswith('positions'):
                in_positions = True
                continue
            if line.lower().startswith('labels'): break
            if in_positions:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        label = parts[0].replace(':', '').upper()
                        coords = [float(x) for x in parts[-3:]]
                        points[label] = np.array(coords)
                    except ValueError: continue
    return points

def align_and_match(source_pts, target_mean):
    """Aligns a single video's points to the master scanner mean using:
    1. Initial Procrustes alignment on landmarks (NAS, LPA, RPA)
    2. ICP refinement using ALL electrodes
    3. Hungarian matching for optimal one-to-one assignment"""
    from scipy.linalg import orthogonal_procrustes
    from scipy.spatial import KDTree
    
    # 1. Initial Landmark Alignment (Procrustes)
    src_lms = np.array([source_pts[l] for l in ["NAS", "LPA", "RPA"] if l in source_pts])
    tgt_lms = np.array([target_mean[l] for l in ["NAS", "LPA", "RPA"] if l in target_mean])
    
    if len(src_lms) < 3: 
        return None
    
    # Procrustes
    mu_s, mu_t = src_lms.mean(0), tgt_lms.mean(0)
    s_c, t_c = src_lms - mu_s, tgt_lms - mu_t
    scale = np.linalg.norm(t_c) / np.linalg.norm(s_c)
    R, _ = orthogonal_procrustes(s_c * scale, t_c)
    
    # Apply initial transform to all pipeline points
    aligned = {}
    for k, v in source_pts.items():
        aligned[k] = (v - mu_s) * scale @ R + mu_t
    
    # 2. ICP Refinement
    scanner_labels = [k for k in target_mean.keys() if k not in LANDMARKS]
    scanner_coords = np.array([target_mean[k] for k in scanner_labels])
    
    pipe_labels = [k for k in aligned.keys() if k not in LANDMARKS]
    pipe_coords = np.array([aligned[k] for k in pipe_labels])
    
    if len(pipe_coords) == 0 or len(scanner_coords) == 0:
        return None
    
    # ICP iterations
    for iteration in range(20):
        tree = KDTree(scanner_coords)
        distances, indices = tree.query(pipe_coords)
        matched_scanner = scanner_coords[indices]
        
        mu_p = pipe_coords.mean(0)
        mu_s = matched_scanner.mean(0)
        
        H = (pipe_coords - mu_p).T @ (matched_scanner - mu_s)
        U, S, Vt = np.linalg.svd(H)
        R_icp = Vt.T @ U.T
        
        if np.linalg.det(R_icp) < 0:
            Vt[-1, :] *= -1
            R_icp = Vt.T @ U.T
        
        pipe_coords_new = (pipe_coords - mu_p) @ R_icp.T + mu_s
        change = np.linalg.norm(pipe_coords_new - pipe_coords)
        pipe_coords = pipe_coords_new
        
        if change < 0.01:
            break
    
    # 3. Hungarian Matching
    n_pipe, n_scan = len(pipe_coords), len(scanner_coords)
    cost_matrix = np.zeros((n_pipe, n_scan))
    for i in range(n_pipe):
        for j in range(n_scan):
            cost_matrix[i, j] = np.linalg.norm(pipe_coords[i] - scanner_coords[j])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match_distances = [(i, j, cost_matrix[i, j]) for i, j in zip(row_ind, col_ind)]
    match_distances.sort(key=lambda x: x[2])
    
    n_to_keep = min(n_scan, len(match_distances))
    good_matches = match_distances[:n_to_keep]
    
    # 4. Build final relabeled dictionary
    relabeled = {}
    for lm in LANDMARKS:
        if lm in aligned:
            relabeled[lm] = aligned[lm]
    
    for i, j, dist in good_matches:
        relabeled[scanner_labels[j]] = pipe_coords[i]
        
    return relabeled

def plot_top_view_head_map(accuracy_errors, electrode_positions, output_dir, title="Map", filename="map.png", vmax=20):
    """Plot electrode errors on a 2D head map - TOP VIEW.
    
    Args:
        accuracy_errors: Dict of {electrode_label: error_value_in_mm}
        electrode_positions: Dict of {electrode_label: 3D_or_2D_position}
                            Used to determine WHERE to plot each electrode
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw head outline
    head_circle = Circle((0, 0), 100, fill=False, color='black', linewidth=2)
    ax.add_patch(head_circle)

    # Draw nose
    nose_x = [0, -10, 10, 0]
    nose_y = [100, 115, 115, 100]
    ax.fill(nose_x, nose_y, color='black')

    # Draw ears
    left_ear = Circle((-108, 0), 8, fill=True, color='black')
    right_ear = Circle((108, 0), 8, fill=True, color='black')
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # Color scale
    vmin = 0
    cmap = plt.cm.RdYlGn_r

    # Plot electrodes
    for label, error in accuracy_errors.items():
        if label in electrode_positions:
            pos = electrode_positions[label]
            # Position convention: pos[1] for x, pos[0] for y
            x_plot = -pos[1]  
            y_plot = pos[0]   

            norm_error = min(error / vmax, 1.0)
            color = cmap(norm_error)
            size = 150

            ax.scatter(x_plot, y_plot, c=[color], s=size, edgecolors='black', linewidths=1, zorder=5)
            ax.annotate(label, (x_plot, y_plot), xytext=(0, -15), textcoords="offset points", 
                        ha='center', fontsize=9, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Error (mm)', fontsize=12)

    ax.text(0, 125, 'Front (NAS)', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, -115, 'Back', ha='center', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=30)

    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def create_clean_labels(file_names, prefix="Video"):
    """Create clean sequential labels for plotting.
    
    Args:
        file_names: List of original file names
        prefix: Prefix for labels ("Video" or "Session")
    
    Returns:
        Dictionary mapping original names to clean labels
    """
    # Sort to ensure consistent ordering
    sorted_names = sorted(file_names)
    label_map = {}
    for i, name in enumerate(sorted_names, start=1):
        label_map[name] = f"{prefix} {i}"
    return label_map

def compute_mean_positions_with_residuals(all_positions_list, file_names):
    """
    Computes the Grand Mean position for each electrode and the specific 
    error/residual for each individual recording session.
    Returns full distribution data for box plots.
    
    MATHEMATICAL PROCESS:
    
    1. For each electrode E:
       - Collect all observations: [P1, P2, ..., PN] (N recordings)
       - Compute mean position: M = (P1 + P2 + ... + PN) / N
    
    2. For each recording i:
       - For each electrode E in that recording:
         - Compute distance: d_i = ||P_i - M||  (Euclidean distance)
       - Average across all electrodes: error_i = mean(all d_i for recording i)
    
    3. Variability for each electrode:
       - mean_distance = mean(||P_i - M||) for all recordings i
       - This is the AVERAGE DEVIATION from mean across recordings
    """
    electrode_positions = defaultdict(list)
    per_file_distances = {name: [] for name in file_names}
    
    # STEP 1: Group all coordinates by their electrode label
    for i, positions_dict in enumerate(all_positions_list):
        fname = file_names[i]
        for label, pos in positions_dict.items():
            electrode_positions[label].append((fname, pos))
    
    # STEP 2: Calculate the Mean (Centroid) for each electrode
    mean_positions = {}
    variability = {}
    for label, observations in electrode_positions.items():
        if observations:
            coords = np.array([obs[1] for obs in observations])  # Extract positions
            m_pos = np.mean(coords, axis=0)  # Mean: (sum of all positions) / N
            mean_positions[label] = m_pos
            
            # Calculate distances from mean
            dists = [np.linalg.norm(c - m_pos) for c in coords]  # ||P_i - M||
            variability[label] = {
                'mean_distance': np.mean(dists),  # Average deviation
                'std_distance': np.std(dists),     # Standard deviation
                'n_recordings': len(observations)
            }
            
    # STEP 3: Calculate how much each specific file deviates from the mean
    for label, observations in electrode_positions.items():
        if label in mean_positions:
            m_pos = mean_positions[label]
            for fname, pos in observations:
                dist = np.linalg.norm(pos - m_pos)  # Distance from mean
                per_file_distances[fname].append(dist)
            
    # STEP 4: Average error across all electrodes for each file
    file_summary = {name: np.mean(d) for name, d in per_file_distances.items() if d}
    
    # Return BOTH summary AND full distribution for box plots
    return mean_positions, variability, file_summary, per_file_distances

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load standard montage for visualization
    standard_montage = parse_elp_montage(STANDARD_MONTAGE_PATH)
    
    # 1. SCANNER DATA
    s_paths = sorted(glob(os.path.join(SCANNER_DIR, "*.elc")))
    s_names = [os.path.basename(f) for f in s_paths]
    s_dicts = [parse_scanner_elc(f) for f in s_paths]
    s_mean, s_variability, s_file_errors, s_distributions = compute_mean_positions_with_residuals(s_dicts, s_names)

    # 2. PIPELINE DATA
    p_paths = sorted(glob(os.path.join(RESULTS_DIR, "*/electrodes_3d.json")))
    p_names = [os.path.basename(os.path.dirname(f)) for f in p_paths]
    p_raw_dicts = [parse_pipeline_json(f) for f in p_paths]
    
    p_matched_dicts = []
    for d in p_raw_dicts:
        matched = align_and_match(d, s_mean)
        if matched: p_matched_dicts.append(matched)
    
    p_mean, p_variability, p_file_errors, p_distributions = compute_mean_positions_with_residuals(p_matched_dicts, p_names)

    # 3. INTER-METHOD ACCURACY
    # Formula: error = ||Pipeline_mean - Scanner_mean|| for each electrode
    common = set(s_mean.keys()) & set(p_mean.keys()) - set(LANDMARKS)
    acc_errors = {l: np.linalg.norm(s_mean[l] - p_mean[l]) for l in common}

    # DECIDE WHICH POSITIONS TO USE FOR PLOTTING
    # Priority: 1) Standard montage, 2) Scanner positions
    if standard_montage:
        print("\n Using standard montage positions for head map visualization")
        plot_positions = standard_montage
    else:
        print("\n Standard montage not found, using scanner positions")
        plot_positions = s_mean

    # --- GENERATE PLOTS ---

    # Plot 1: Scanner Reliability BOX PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create clean labels
    s_label_map = create_clean_labels(list(s_distributions.keys()), prefix="Session")
    s_sorted_names = sorted(s_distributions.keys())
    s_clean_labels = [s_label_map[name] for name in s_sorted_names]
    s_data = [s_distributions[name] for name in s_sorted_names]
    
    bp = ax.boxplot(s_data, labels=s_clean_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
    
    ax.set_title("Scanner Reliability (Internal Consistency)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Deviation from mean (mm)", fontsize=12)
    ax.set_xlabel("Recording Session", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add visual line annotation with symbols
    ax.plot([0.82, 0.87], [0.96, 0.96], transform=ax.transAxes, 
            color='red', linewidth=2, solid_capstyle='round')
    ax.text(0.88, 0.96, '= Median', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    ax.plot([0.82, 0.87], [0.91, 0.91], transform=ax.transAxes, 
            color='blue', linewidth=2, linestyle='--', solid_capstyle='round')
    ax.text(0.88, 0.91, '= Mean', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scanner_reliability_boxplot.png'), dpi=150)
    plt.close()
    print(" Scanner reliability box plot saved")

    # Plot 2: Pipeline Reliability BOX PLOT
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create clean labels
    p_label_map = create_clean_labels(list(p_distributions.keys()), prefix="Video")
    p_sorted_names = sorted(p_distributions.keys())
    p_clean_labels = [p_label_map[name] for name in p_sorted_names]
    p_data = [p_distributions[name] for name in p_sorted_names]
    
    bp = ax.boxplot(p_data, labels=p_clean_labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='darkred', markersize=6, alpha=0.5))
    
    ax.set_title("Pipeline Reliability (Internal Consistency)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Deviation from mean (mm)", fontsize=12)
    ax.set_xlabel("Video Recording", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add visual line annotation with symbols
    ax.plot([0.82, 0.87], [0.96, 0.96], transform=ax.transAxes, 
            color='darkred', linewidth=2, solid_capstyle='round')
    ax.text(0.88, 0.96, '= Median', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    ax.plot([0.82, 0.87], [0.91, 0.91], transform=ax.transAxes, 
            color='blue', linewidth=2, linestyle='--', solid_capstyle='round')
    ax.text(0.88, 0.91, '= Mean', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pipeline_reliability_boxplot.png'), dpi=150)
    plt.close()
    print("Pipeline reliability box plot saved")

    # Plot 3: Method Comparison - Combined BOX PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_scanner_errors = []
    for dist_list in s_distributions.values():
        all_scanner_errors.extend(dist_list)
    
    all_pipeline_errors = []
    for dist_list in p_distributions.values():
        all_pipeline_errors.extend(dist_list)
    

    
    bp = ax.boxplot([all_scanner_errors, all_pipeline_errors], 
                    labels=['Scanner (Digitizer)', 'Pipeline (Video)'],
                    patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=6, alpha=0.5))
    
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_ylabel('Deviation from Mean (mm)', fontsize=12)
    ax.set_title('Repeatability Comparison: Scanner vs Pipeline', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    

    # Add visual line annotation with symbols (left side)
    ax.plot([0.02, 0.07], [0.96, 0.96], transform=ax.transAxes, 
            color='red', linewidth=2, solid_capstyle='round')
    ax.text(0.08, 0.96, '= Median', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    ax.plot([0.02, 0.07], [0.91, 0.91], transform=ax.transAxes, 
            color='blue', linewidth=2, linestyle='--', solid_capstyle='round')
    ax.text(0.08, 0.91, '= Mean', transform=ax.transAxes, 
            fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'repeatability_comparison.png'), dpi=150)
    plt.close()
    print(" Repeatability comparison saved")

    # Plot 4: Accuracy Head Map (use standard montage or scanner positions)
    plot_top_view_head_map(acc_errors, plot_positions, OUTPUT_DIR, 
                           title="Inter-Method Accuracy (Pipeline Mean vs Scanner Mean)", 
                           filename="accuracy_head_map.png")
    print(" Accuracy head map saved")

    # Plot 5: Pipeline Variability Head Map (use standard montage or scanner positions)
    p_vars = {k: v['mean_distance'] for k, v in p_variability.items() if k not in LANDMARKS}
    plot_top_view_head_map(p_vars, plot_positions, OUTPUT_DIR, 
                           title="Pipeline Internal Variability", 
                           filename="pipeline_variability_map.png")
    print(" Pipeline variability map saved")

    # Plot 6: Scanner Variability Head Map (use standard montage or scanner positions)
    s_vars = {k: v['mean_distance'] for k, v in s_variability.items() if k not in LANDMARKS}
    plot_top_view_head_map(s_vars, plot_positions, OUTPUT_DIR, 
                           title="Scanner Internal Variability", 
                           filename="scanner_variability_map.png",
                           vmax=5)
    print(" Scanner variability map saved")

    # Print numerical summary

    print("ERROR CALCULATION SUMMARY")
    
    print("\n1. SCANNER REPEATABILITY (Internal Consistency)")
    print("   Formula: For each session, average deviation of all electrodes from mean")
    print(f"   Mean across sessions: {np.mean(list(s_file_errors.values())):.2f} mm")
    print(f"   Std Dev: {np.std(list(s_file_errors.values())):.2f} mm")
    print(f"   Median: {np.median([e for dist in s_distributions.values() for e in dist]):.2f} mm")
    print(f"   Range: {np.min([e for dist in s_distributions.values() for e in dist]):.2f} - {np.max([e for dist in s_distributions.values() for e in dist]):.2f} mm")
    
    print("\n2. PIPELINE REPEATABILITY (Internal Consistency)")
    print("   Formula: For each video, average deviation of all electrodes from mean")
    print(f"   Mean across videos: {np.mean(list(p_file_errors.values())):.2f} mm")
    print(f"   Std Dev: {np.std(list(p_file_errors.values())):.2f} mm")
    print(f"   Median: {np.median([e for dist in p_distributions.values() for e in dist]):.2f} mm")
    print(f"   Range: {np.min([e for dist in p_distributions.values() for e in dist]):.2f} - {np.max([e for dist in p_distributions.values() for e in dist]):.2f} mm")
    
    print("\n3. INTER-METHOD ACCURACY (Pipeline vs Scanner)")
    print("   Formula: For each electrode, ||Pipeline_mean - Scanner_mean||")
    print(f"   Mean: {np.mean(list(acc_errors.values())):.2f} mm")
    print(f"   Std Dev: {np.std(list(acc_errors.values())):.2f} mm")
    print(f"   Median: {np.median(list(acc_errors.values())):.2f} mm")
    print(f"   Range: {np.min(list(acc_errors.values())):.2f} - {np.max(list(acc_errors.values())):.2f} mm")
    print(f"   Electrodes < 5mm: {100*sum(1 for e in acc_errors.values() if e < 5)/len(acc_errors):.1f}%")
    print(f"   Electrodes < 10mm: {100*sum(1 for e in acc_errors.values() if e < 10)/len(acc_errors):.1f}%")
    print(f"   Electrodes < 15mm: {100*sum(1 for e in acc_errors.values() if e < 15)/len(acc_errors):.1f}%")
    
    if standard_montage:
        print(f"Visualization: Using STANDARD MONTAGE positions from {os.path.basename(STANDARD_MONTAGE_PATH)}")
    else:
        print(f"Visualization: Using SCANNER positions")
    print(f"All plots saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()