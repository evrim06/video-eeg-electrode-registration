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

def compute_mean_with_residuals(positions_list, names):
    """Computes grand mean and how much each individual file deviates from it."""
    electrode_positions = defaultdict(list)
    per_file_distances = {name: [] for name in names}
    
    for i, pos_dict in enumerate(positions_list):
        for label, coords in pos_dict.items():
            electrode_positions[label].append(coords)
    
    mean_positions = {l: np.mean(p, axis=0) for l, p in electrode_positions.items()}
    
    for i, pos_dict in enumerate(positions_list):
        fname = names[i]
        for label, coords in pos_dict.items():
            dist = np.linalg.norm(coords - mean_positions[label])
            per_file_distances[fname].append(dist)
            
    file_errors = {n: np.mean(d) for n, d in per_file_distances.items() if d}
    return mean_positions, file_errors

def align_and_match(source_pts, target_mean):
    """Aligns a single video's points to the master scanner mean using:
    1. Initial Procrustes alignment on landmarks (NAS, LPA, RPA)
    2. ICP refinement using ALL electrodes
    3. Hungarian matching for optimal one-to-one assignment
    Automatically filters out extra electrodes not in scanner data."""
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
    
    # 2. ICP Refinement (using all electrodes, not just landmarks)
    scanner_labels = [k for k in target_mean.keys() if k not in LANDMARKS]
    scanner_coords = np.array([target_mean[k] for k in scanner_labels])
    
    pipe_labels = [k for k in aligned.keys() if k not in LANDMARKS]
    pipe_coords = np.array([aligned[k] for k in pipe_labels])
    
    if len(pipe_coords) == 0 or len(scanner_coords) == 0:
        return None
    
    # ICP iterations
    for iteration in range(20):
        # Match each pipeline point to nearest scanner point
        tree = KDTree(scanner_coords)
        distances, indices = tree.query(pipe_coords)
        
        # Get matched scanner points
        matched_scanner = scanner_coords[indices]
        
        # Compute centroids
        mu_p = pipe_coords.mean(0)
        mu_s = matched_scanner.mean(0)
        
        # Compute optimal rotation (Kabsch algorithm)
        H = (pipe_coords - mu_p).T @ (matched_scanner - mu_s)
        U, S, Vt = np.linalg.svd(H)
        R_icp = Vt.T @ U.T
        
        # Ensure proper rotation (not reflection)
        if np.linalg.det(R_icp) < 0:
            Vt[-1, :] *= -1
            R_icp = Vt.T @ U.T
        
        # Apply transform
        pipe_coords_new = (pipe_coords - mu_p) @ R_icp.T + mu_s
        
        # Check convergence
        change = np.linalg.norm(pipe_coords_new - pipe_coords)
        pipe_coords = pipe_coords_new
        
        if change < 0.01:  # Converged (less than 0.01mm change)
            break
    
    # 3. Hungarian Matching (optimal one-to-one assignment)
    n_pipe, n_scan = len(pipe_coords), len(scanner_coords)
    cost_matrix = np.zeros((n_pipe, n_scan))
    for i in range(n_pipe):
        for j in range(n_scan):
            cost_matrix[i, j] = np.linalg.norm(pipe_coords[i] - scanner_coords[j])
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Sort by match quality and keep only best N (N = scanner count)
    match_distances = [(i, j, cost_matrix[i, j]) for i, j in zip(row_ind, col_ind)]
    match_distances.sort(key=lambda x: x[2])
    
    n_to_keep = min(n_scan, len(match_distances))
    good_matches = match_distances[:n_to_keep]
    rejected = match_distances[n_to_keep:]
    
    if rejected:
        for i, j, dist in rejected:
            print(f"  Excluded extra electrode: {pipe_labels[i]} ({dist:.1f}mm from {scanner_labels[j]})")
    
    # 4. Build final relabeled dictionary
    relabeled = {}
    
    # Add landmarks (also need ICP transform applied)
    for lm in LANDMARKS:
        if lm in aligned:
            # Apply same ICP transform to landmarks
            lm_pos = aligned[lm]
            # We need to track cumulative transform - simplified: use original aligned
            relabeled[lm] = aligned[lm]
    
    # Add matched electrodes with ICP-refined positions
    for i, j, dist in good_matches:
        relabeled[scanner_labels[j]] = pipe_coords[i]
        
    return relabeled

def plot_top_view_head_map(accuracy_errors, scanner_mean, output_dir, title="Map", filename="map.png", vmax=20):
    """Plot electrode errors on a 2D head map - TOP VIEW."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Draw head outline
    head_circle = Circle((0, 0), 100, fill=False, color='black', linewidth=2)
    ax.add_patch(head_circle)

    # 2. Draw nose
    nose_x = [0, -10, 10, 0]
    nose_y = [100, 115, 115, 100]
    ax.fill(nose_x, nose_y, color='black')

    # 3. Draw ears
    left_ear = Circle((-108, 0), 8, fill=True, color='black')
    right_ear = Circle((108, 0), 8, fill=True, color='black')
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 4. Color scale
    vmin = 0
    cmap = plt.cm.RdYlGn_r

    # 5. Plot electrodes
    for label, error in accuracy_errors.items():
        if label in scanner_mean:
            pos = scanner_mean[label]
            x_plot = -pos[1]  
            y_plot = pos[0]   

            norm_error = min(error / vmax, 1.0)
            color = cmap(norm_error)
            size = 150 + (error * 10) 

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
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_variability_map(variability_dict, mean_positions, title, filename, output_dir):
    """Plots internal jitter (repeatability) on a head map."""
    # This uses your existing head map drawing logic
    # But instead of Accuracy Error, it plots the 'mean_distance' from compute_mean
    errors = {k: v['mean_distance'] for k, v in variability_dict.items() if k not in LANDMARKS}
    plot_top_view_head_map(errors, mean_positions, output_dir, title=title, filename=filename, vmax=10)

def compute_mean_positions_with_residuals(all_positions_list, file_names):
    """
    Computes the Grand Mean position for each electrode and the specific 
    error/residual for each individual recording session.
    """
    from collections import defaultdict
    import numpy as np

    electrode_positions = defaultdict(list)
    per_file_distances = {name: [] for name in file_names}
    
    # 1. Group all coordinates by their electrode label
    for i, positions_dict in enumerate(all_positions_list):
        fname = file_names[i]
        for label, pos in positions_dict.items():
            electrode_positions[label].append((fname, pos))
    
    # 2. Calculate the Mean (Centroid) for each electrode
    mean_positions = {}
    variability = {}
    for label, observations in electrode_positions.items():
        if observations:
            coords = np.array([obs[1] for obs in observations])
            m_pos = np.mean(coords, axis=0)
            mean_positions[label] = m_pos
            
            # Calculate standard variability for the variability head-map
            dists = [np.linalg.norm(c - m_pos) for c in coords]
            variability[label] = {
                'mean_distance': np.mean(dists),
                'n_recordings': len(observations)
            }
            
    # 3. Calculate how much each specific file deviates from the mean
    for label, observations in electrode_positions.items():
        if label in mean_positions:
            m_pos = mean_positions[label]
            for fname, pos in observations:
                dist = np.linalg.norm(pos - m_pos)
                per_file_distances[fname].append(dist)
            
    # 4. Average the error across all electrodes for each file
    file_summary = {name: np.mean(d) for name, d in per_file_distances.items() if d}
    
    return mean_positions, variability, file_summary

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. SCANNER DATA
    s_paths = sorted(glob(os.path.join(SCANNER_DIR, "*.elc")))
    s_names = [os.path.basename(f) for f in s_paths]
    s_dicts = [parse_scanner_elc(f) for f in s_paths]
    s_mean, s_variability, s_file_errors = compute_mean_positions_with_residuals(s_dicts, s_names)

    # 2. PIPELINE DATA
    p_paths = sorted(glob(os.path.join(RESULTS_DIR, "*/electrodes_3d.json")))
    p_names = [os.path.basename(os.path.dirname(f)) for f in p_paths]
    p_raw_dicts = [parse_pipeline_json(f) for f in p_paths]
    
    p_matched_dicts = []
    for d in p_raw_dicts:
        matched = align_and_match(d, s_mean) # Snap to scanner labels
        if matched: p_matched_dicts.append(matched)
    
    p_mean, p_variability, p_file_errors = compute_mean_positions_with_residuals(p_matched_dicts, p_names)

    # 3. INTER-METHOD ACCURACY
    common = set(s_mean.keys()) & set(p_mean.keys()) - set(LANDMARKS)
    acc_errors = {l: np.linalg.norm(s_mean[l] - p_mean[l]) for l in common}

    # --- GENERATE ALL 5 PLOTS ---

    # Plot 1: Scanner Reliability (Bar chart of sessions)
    plt.figure(figsize=(10, 5))
    plt.bar(s_file_errors.keys(), s_file_errors.values(), color='#3498db', edgecolor='black')
    plt.title("Scanner Reliability (Internal Consistency)")
    plt.ylabel("Avg deviation from mean (mm)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scanner_reliability_bars.png'))

    # Plot 2: Pipeline Reliability (Bar chart of sessions)
    plt.figure(figsize=(10, 5))
    plt.bar(p_file_errors.keys(), p_file_errors.values(), color='#e74c3c', edgecolor='black')
    plt.title("Pipeline Reliability (Internal Consistency)")
    plt.ylabel("Avg deviation from mean (mm)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pipeline_reliability_bars.png'))

    # Plot 3: Method Comparison (Precision)
    s_rep = np.mean(list(s_file_errors.values()))
    p_rep = np.mean(list(p_file_errors.values()))
    plt.figure(figsize=(6, 6))
    plt.bar(['Scanner', 'Pipeline'], [s_rep, p_rep], color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.ylabel('Mean Repeatability Error (mm)')
    plt.title('Precision Comparison')
    plt.savefig(os.path.join(OUTPUT_DIR, 'repeatability_comparison.png'))

    # Plot 4: Final Accuracy Head Map
    plot_top_view_head_map(acc_errors, s_mean, OUTPUT_DIR, 
                           title="Inter-Method Accuracy\n(Pipeline Mean vs Scanner Mean)", 
                           filename="accuracy_head_map.png")

    # Plot 5: Pipeline Variability Head Map
    p_vars = {k: v['mean_distance'] for k, v in p_variability.items() if k not in LANDMARKS}
    plot_top_view_head_map(p_vars, p_mean, OUTPUT_DIR, 
                           title="Pipeline Internal Variability\n(Where is the jitter?)", 
                           filename="pipeline_variability_map.png")

    print(f"All 5 validation plots saved to {OUTPUT_DIR}")
    # Plot 6: Scanner Variability Head Map (NEW)
    s_vars = {k: v['mean_distance'] for k, v in s_variability.items() if k not in LANDMARKS}
    plot_top_view_head_map(s_vars, s_mean, OUTPUT_DIR, 
                           title="Scanner Internal Variability\n(Precision of the Digitizer)", 
                           filename="scanner_variability_map.png",
                           vmax=5) # Lower vmax because scanner should be very precise
    # Print numerical summary
    print("SUMMARY")
    print(f"\nScanner repeatability: {np.mean(list(s_file_errors.values())):.2f} ± {np.std(list(s_file_errors.values())):.2f} mm")
    print(f"Pipeline repeatability: {np.mean(list(p_file_errors.values())):.2f} ± {np.std(list(p_file_errors.values())):.2f} mm")
    print(f"\nAccuracy (Pipeline Mean vs Scanner Mean):")
    print(f"  Mean: {np.mean(list(acc_errors.values())):.2f} ± {np.std(list(acc_errors.values())):.2f} mm")
    print(f"  Median: {np.median(list(acc_errors.values())):.2f} mm")
    print(f"  LA < 5mm: {100*sum(1 for e in acc_errors.values() if e < 5)/len(acc_errors):.1f}%")
    print(f"  LA < 10mm: {100*sum(1 for e in acc_errors.values() if e < 10)/len(acc_errors):.1f}%")

if __name__ == "__main__":
    main()
