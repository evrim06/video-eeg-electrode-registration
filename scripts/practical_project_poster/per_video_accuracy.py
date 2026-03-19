"""
PER-VIDEO ACCURACY
Compares each individual pipeline recording against the digitizer grand mean.
Produces per_video_accuracy.png — one boxplot column per video.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from glob import glob
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.linalg import orthogonal_procrustes

# ============================================================
# CONFIGURATION  — adjust paths to match your project layout
# ============================================================
RESULTS_DIR   = "results"
SCANNER_DIR   = "Scanner_recordings"
OUTPUT_DIR    = "validation_results"
LANDMARKS     = ["NAS", "LPA", "RPA", "INION"]

COLOR_PIPELINE = '#DA91BA'   # UOL Pantone 204
PLOT_DPI       = 1200
SAVE_VECTOR    = True


def _save_plot(fig, output_dir, filename):
    fig.savefig(os.path.join(output_dir, filename), dpi=PLOT_DPI, bbox_inches='tight')

# ============================================================
# PARSERS
# ============================================================

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
    if not os.path.exists(path):
        return points
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('positions'):
                in_positions = True; continue
            if line.lower().startswith('labels'):
                break
            if in_positions:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        points[parts[0].replace(':', '').upper()] = np.array(
                            [float(x) for x in parts[-3:]])
                    except ValueError:
                        continue
    return points


# ============================================================
# ALIGNMENT & MATCHING  (identical to main script)
# ============================================================

def align_and_match(source_pts, target_mean):
    src_lms = np.array([source_pts[l] for l in ["NAS", "LPA", "RPA"] if l in source_pts])
    tgt_lms = np.array([target_mean[l] for l in ["NAS", "LPA", "RPA"] if l in target_mean])
    if len(src_lms) < 3:
        return None
    mu_s, mu_t = src_lms.mean(0), tgt_lms.mean(0)
    s_c, t_c   = src_lms - mu_s, tgt_lms - mu_t
    scale = np.linalg.norm(t_c) / np.linalg.norm(s_c)
    R, _  = orthogonal_procrustes(s_c * scale, t_c)
    aligned = {k: (v - mu_s) * scale @ R + mu_t for k, v in source_pts.items()}

    scanner_labels = [k for k in target_mean if k not in LANDMARKS]
    scanner_coords = np.array([target_mean[k] for k in scanner_labels])
    pipe_labels    = [k for k in aligned     if k not in LANDMARKS]
    pipe_coords    = np.array([aligned[k]    for k in pipe_labels])
    if len(pipe_coords) == 0 or len(scanner_coords) == 0:
        return None

    for _ in range(20):
        tree = cKDTree(scanner_coords)
        _, indices = tree.query(pipe_coords)
        matched = scanner_coords[indices]
        mu_p, mu_m = pipe_coords.mean(0), matched.mean(0)
        H = (pipe_coords - mu_p).T @ (matched - mu_m)
        U, S, Vt = np.linalg.svd(H)
        R_icp = Vt.T @ U.T
        if np.linalg.det(R_icp) < 0:
            Vt[-1, :] *= -1; R_icp = Vt.T @ U.T
        new_coords = (pipe_coords - mu_p) @ R_icp.T + mu_m
        if np.linalg.norm(new_coords - pipe_coords) < 0.01:
            break
        pipe_coords = new_coords

    cost = np.array([[np.linalg.norm(pipe_coords[i] - scanner_coords[j])
                      for j in range(len(scanner_coords))]
                     for i in range(len(pipe_coords))])
    row_ind, col_ind = linear_sum_assignment(cost)
    relabeled = {lm: aligned[lm] for lm in LANDMARKS if lm in aligned}
    for i, j in zip(row_ind, col_ind):
        relabeled[scanner_labels[j]] = pipe_coords[i]
    return relabeled


# ============================================================
# HELPERS
# ============================================================

def compute_scanner_mean(s_dicts):
    electrode_positions = defaultdict(list)
    for d in s_dicts:
        for label, pos in d.items():
            electrode_positions[label].append(pos)
    return {label: np.mean(coords, axis=0)
            for label, coords in electrode_positions.items()}


def numeric_sort_key(name):
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else 0


def create_label_map(names, prefix="Video"):
    sorted_names = sorted(names, key=numeric_sort_key)
    return ({n: f"{prefix} {i}" for i, n in enumerate(sorted_names, 1)},
            {n: i               for i, n in enumerate(sorted_names, 1)})


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Scanner grand mean ---
    s_paths = sorted(glob(os.path.join(SCANNER_DIR, "*.elc")))
    s_dicts = [parse_scanner_elc(f) for f in s_paths]
    s_mean  = compute_scanner_mean(s_dicts)

    # --- Pipeline: align every recording to scanner mean ---
    p_paths = sorted(glob(os.path.join(RESULTS_DIR, "*/electrodes_3d.json")))
    p_names = [os.path.basename(os.path.dirname(f)) for f in p_paths]
    p_raw   = [parse_pipeline_json(f) for f in p_paths]

    p_aligned, p_aligned_names = [], []
    for i, d in enumerate(p_raw):
        m = align_and_match(d, s_mean)
        if m:
            p_aligned.append(m)
            p_aligned_names.append(p_names[i])

    label_map, index_map = create_label_map(p_aligned_names)

    # Per-video electrode errors vs digitizer grand mean
    electrode_labels = sorted(set(s_mean) - set(LANDMARKS))
    video_errors = {}
    for name, aligned in zip(p_aligned_names, p_aligned):
        errs = [np.linalg.norm(aligned[l] - s_mean[l])
                for l in electrode_labels if l in aligned]
        if errs:
            video_errors[name] = errs

    # --- Console summary ---
    print(f"\n{'=' * 72}")
    print("PER-VIDEO ACCURACY  (each recording vs. digitizer grand mean)")
    print(f"{'=' * 72}")
    header = (f"  {'Video':>10s}  {'Mean':>8s}  {'Median':>8s}  "
              f"{'Min':>7s}  {'Max':>7s}  {'<5mm':>7s}  {'<10mm':>7s}")
    print(header)
    print(f"  {'─' * 65}")
    sorted_names = sorted(p_aligned_names, key=numeric_sort_key)
    for name in sorted_names:
        errs = video_errors.get(name, [])
        if not errs:
            continue
        pct5  = 100 * sum(1 for e in errs if e < 5)  / len(errs)
        pct10 = 100 * sum(1 for e in errs if e < 10) / len(errs)
        print(f"  {label_map[name]:>10s}  {np.mean(errs):8.2f}  "
              f"{np.median(errs):8.2f}  {np.min(errs):7.2f}  "
              f"{np.max(errs):7.2f}  {pct5:6.1f}%  {pct10:6.1f}%")
    all_errs = [e for v in video_errors.values() for e in v]
    print(f"  {'─' * 65}")
    pct5  = 100 * sum(1 for e in all_errs if e < 5)  / len(all_errs)
    pct10 = 100 * sum(1 for e in all_errs if e < 10) / len(all_errs)
    print(f"  {'Overall':>10s}  {np.mean(all_errs):8.2f}  "
          f"{np.median(all_errs):8.2f}  {np.min(all_errs):7.2f}  "
          f"{np.max(all_errs):7.2f}  {pct5:6.1f}%  {pct10:6.1f}%")

    # ============================================================
    # PLOT: per-video boxplots  (no mean annotation at bottom)
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    data   = [video_errors[n] for n in sorted_names if n in video_errors]
    labels = [label_map[n]    for n in sorted_names if n in video_errors]

    bp = ax.boxplot(data, labels=labels,
                    patch_artist=True, showmeans=True, meanline=True,
                    showfliers=False,
                    boxprops      =dict(facecolor=COLOR_PIPELINE, alpha=0.6),
                    medianprops   =dict(color='#B0608F', linewidth=2),
                    meanprops     =dict(color='blue',    linewidth=2, linestyle='--'),
                    whiskerprops  =dict(linewidth=1.5),
                    capprops      =dict(linewidth=1.5))

    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        ax.scatter(rng.normal(i + 1, 0.06, len(d)), d,
                   alpha=0.5, s=25, color='#A05A8A',
                   edgecolors='black', linewidth=0.3, zorder=3)

    # Legend: median / mean lines + scatter dot
    legend_elements = [
        Line2D([0], [0], color='#B0608F', linewidth=2,         label='Median'),
        Line2D([0], [0], color='blue',    linewidth=2,
               linestyle='--',                                  label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#A05A8A',
               markersize=7, markeredgecolor='black',
               markeredgewidth=0.5,                             label='One electrode'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=9, framealpha=0.85)

    ax.set_xlabel('Video Recording', fontsize=16)
    ax.set_ylabel('Euclidean distance to digitizer mean (mm)', fontsize=16)
    ax.set_title('Inter-Method Accuracy — Each Video vs Digitizer Mean',
                 fontsize=18, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, OUTPUT_DIR, 'per_video_accuracy.png')
    plt.close()
    print(f"\n  Saved: per_video_accuracy.png  ({OUTPUT_DIR}/)")


if __name__ == "__main__":
    main()