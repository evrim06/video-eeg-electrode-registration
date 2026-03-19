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
import pandas as pd
import pingouin as pg
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR = "results"
SCANNER_DIR = "Scanner_recordings"
OUTPUT_DIR = "validation_results"
LANDMARKS = ["NAS", "LPA", "RPA", "INION"]
STANDARD_MONTAGE_PATH = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\mobile24.elp"

# === UOL Oldenburg Official Palette ===
COLOR_SCANNER  = '#00ABD9'   # Akzentfarben Blau 2  — Digitizer
COLOR_PIPELINE = '#666699'   # Akzentfarben 
COLOR_ACCURACY = '#EE7100'   # Akzentfarben Orange 2 — Inter-method

# --- Quality: 1200 DPI for screen projection ---
PLOT_DPI   = 1200       

LABEL_OFFSETS = {
    'TP9':  (-18, -12), 'C3':  (0, -15),
    'TP10': ( 18, -12), 'C4':  (0, -15),
    'FP1':  (-12,  12), 'FP2': (12,  12),
}

def _save_plot(fig, output_dir, filename):
    fig.savefig(os.path.join(output_dir, filename), dpi=PLOT_DPI, bbox_inches='tight')


# ============================================================
# DATA PARSING
# ============================================================

def parse_elp_montage(elp_path):
    if not os.path.exists(elp_path):
        return None

    raw = {}
    with open(elp_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == 'EEG':
                label = parts[1].upper()
                try:
                    polar_deg = float(parts[2])
                    azimuth_deg = float(parts[3])
                    
                    # Handle negative polar angles by flipping the azimuth
                    if polar_deg < 0:
                        polar_deg = abs(polar_deg)
                        azimuth_deg += 180.0
                    
                    # --- THE "GLOBE" PROJECTION ---
                    # 1. Convert angle to a fraction (90 degrees = 1.0 = equator)
                    r_frac = polar_deg / 90.0
                    
                    # 2. Apply a curve to push inner electrodes outward
                    r_proj = np.power(r_frac, 0.6)
                    
                    # 3. Scale to fit the drawn head (radius 95)
                    r = r_proj * 95.0
                    
                    azimuth_rad = np.radians(azimuth_deg)
                    x = r * np.cos(azimuth_rad)
                    y = r * np.sin(azimuth_rad)
                    
                    # Format for downstream plotting (pos[0]=Y, pos[1]=-X)
                    raw[label] = np.array([y, -x, float(parts[4])])
                except ValueError:
                    continue

    return raw if raw else None


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
                in_positions = True
                continue
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
# ALIGNMENT & MATCHING
# ============================================================

def align_and_match(source_pts, target_mean):
    from scipy.linalg import orthogonal_procrustes
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
            Vt[-1, :] *= -1
            R_icp = Vt.T @ U.T
        new_coords = (pipe_coords - mu_p) @ R_icp.T + mu_m
        if np.linalg.norm(new_coords - pipe_coords) < 0.01:
            break
        pipe_coords = new_coords

    cost = np.array([[np.linalg.norm(pipe_coords[i] - scanner_coords[j])
                      for j in range(len(scanner_coords))]
                     for i in range(len(pipe_coords))])
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = sorted([(i, j, cost[i, j]) for i, j in zip(row_ind, col_ind)],
                     key=lambda x: x[2])
    relabeled = {lm: aligned[lm] for lm in LANDMARKS if lm in aligned}
    for i, j, _ in matches[:len(scanner_labels)]:
        relabeled[scanner_labels[j]] = pipe_coords[i]
    return relabeled


# ============================================================
# STATISTICS HELPERS
# ============================================================

def compute_mean_positions_with_residuals(all_positions_list, file_names):
    """
    Returns:
        mean_positions       – {label: 3d array}
        variability          – {label: {mean_distance, std_distance, n_recordings}}
        file_summary         – {filename: mean_deviation_across_electrodes}
        per_file_distances   – {filename: [deviation_per_electrode]}   (order = electrode_positions keys)
        per_elec_distances   – {electrode: [deviation_per_recording]}  ← NEW, needed for labeled scatter
    """
    electrode_positions = defaultdict(list)
    per_file_distances  = {name: [] for name in file_names}

    for i, pos_dict in enumerate(all_positions_list):
        fname = file_names[i]
        for label, pos in pos_dict.items():
            electrode_positions[label].append((fname, pos))

    mean_positions, variability = {}, {}
    for label, obs in electrode_positions.items():
        if obs:
            coords = np.array([o[1] for o in obs])
            m = np.mean(coords, axis=0)
            mean_positions[label] = m
            dists = [np.linalg.norm(c - m) for c in coords]
            variability[label] = {
                'mean_distance':  np.mean(dists),
                'std_distance':   np.std(dists),
                'n_recordings':   len(obs),
            }

    # per-file (for recording-level boxplot) AND per-electrode (for electrode-level boxplot)
    per_elec_distances = defaultdict(list)
    for label, obs in electrode_positions.items():
        if label in mean_positions:
            m = mean_positions[label]
            for fname, pos in obs:
                d = np.linalg.norm(pos - m)
                per_file_distances[fname].append(d)
                per_elec_distances[label].append(d)

    file_summary = {n: np.mean(d) for n, d in per_file_distances.items() if d}
    return mean_positions, variability, file_summary, per_file_distances, dict(per_elec_distances)


def find_poor_reconstructions(file_errors, label_map):
    ranked    = sorted(file_errors.items(), key=lambda x: x[1])
    vals      = [v for _, v in ranked]
    max_gap   = np.argmax(np.diff(vals))
    threshold = (vals[max_gap] + vals[max_gap + 1]) / 2
    return [n for n, v in file_errors.items() if v > threshold]


def create_clean_labels(file_names, prefix="Video"):
    sorted_names = sorted(file_names, key=lambda n: int(''.join(filter(str.isdigit, n)) or 0))
    return (
        {n: f"{prefix} {i}" for i, n in enumerate(sorted_names, 1)},
        {n: i               for i, n in enumerate(sorted_names, 1)},
    )


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _draw_head(ax):
    gray = '#777777'
    ax.add_patch(Circle((0, 0), 100, fill=False, color=gray, linewidth=2.5))
    ax.fill([0, -10, 10, 0], [100, 115, 115, 100], color=gray)
    ax.add_patch(Circle((-108, 0), 8, fill=True, color=gray))
    ax.add_patch(Circle(( 108, 0), 8, fill=True, color=gray))
    ax.text(-108, -15, 'LPA', ha='center', fontsize=9, fontweight='bold', color=gray)
    ax.text( 108, -15, 'RPA', ha='center', fontsize=9, fontweight='bold', color=gray)


def _plot_electrodes_on_ax(ax, errors, positions, cmap, vmax):
    for label, error in errors.items():
        if label in positions:
            pos    = positions[label]
            color  = cmap(min(error / vmax, 1.0))
            ax.scatter(-pos[1], pos[0], c=[color], s=180,
                       edgecolors='black', linewidths=1.5, zorder=5)
            offset = LABEL_OFFSETS.get(label, (0, -15))
            ax.annotate(label, (-pos[1], pos[0]), xytext=offset,
                        textcoords="offset points", ha='center',
                        fontsize=9, fontweight='bold')


def _legend_lines(ax, y_positions=(0.96, 0.91), x_left=0.70, x_right=0.78, x_text=0.81):
    ax.plot([x_left, x_right], [y_positions[0]] * 2,
            transform=ax.transAxes, color='red', linewidth=2)
    ax.text(x_text, y_positions[0], '= Median',
            transform=ax.transAxes, fontsize=9, va='center')
    ax.plot([x_left, x_right], [y_positions[1]] * 2,
            transform=ax.transAxes, color='blue', linewidth=2, linestyle='--')
    ax.text(x_text, y_positions[1], '= Mean',
            transform=ax.transAxes, fontsize=9, va='center')


# ============================================================
# BOXPLOT — ACCURACY
# ============================================================

def plot_accuracy_boxplot(acc_clean, output_dir):
    data = list(acc_clean.values())

    fig, ax = plt.subplots(figsize=(5, 7))

    bp = ax.boxplot([data],
                    tick_labels=[f'Smartphone Pipeline\n(n={len(data)} electrodes)'],
                    patch_artist=True, showmeans=True, meanline=True,
                    showfliers=False, widths=0.5,
                    medianprops=dict(color='red',  linewidth=2),
                    meanprops  =dict(color='blue', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops    =dict(linewidth=1.5))
    bp['boxes'][0].set(facecolor=COLOR_ACCURACY, alpha=0.6)

    rng = np.random.default_rng(42)
    sc  = ax.scatter(rng.normal(1, 0.04, len(data)), data,
                     alpha=0.8, s=60, color='#C55A00',
                     edgecolors='black', linewidth=0.5, zorder=4,
                     label='One electrode')

    mean_val = np.mean(data)
    ax.annotate(f'{mean_val:.1f} mm', xy=(1, mean_val),
                xytext=(40, 0), textcoords='offset points',
                fontsize=12, fontweight='bold', va='center')

    ax.legend(handles=[sc], loc='upper right', fontsize=9, framealpha=0.8)
    ax.set_ylabel('Euclidean distance to digitizer mean (mm)', fontsize=12)
    ax.set_title('Inter-Method Accuracy\n(Pipeline vs. Digitizer Mean)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _legend_lines(ax)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'accuracy_boxplot.png')
    plt.close()


# ============================================================
# BOXPLOT — RECORDING-LEVEL REPEATABILITY
# ============================================================

def plot_poster_precision_boxplot(s_file_errors, p_errors_clean, output_dir):
    s_vals       = list(s_file_errors.values())
    p_vals_clean = list(p_errors_clean.values())

    fig, ax = plt.subplots(figsize=(6, 6))

    bp = ax.boxplot(
        [s_vals, p_vals_clean],
        tick_labels=['Digitizer\n(Ground Truth)', 'Smartphone Pipeline'],
        patch_artist=True, showmeans=True, meanline=True,
        showfliers=False, widths=0.5,
        medianprops=dict(color='red',  linewidth=2),
        meanprops  =dict(color='blue', linewidth=2, linestyle='--'),
        whiskerprops=dict(linewidth=1.5),
        capprops    =dict(linewidth=1.5))
    bp['boxes'][0].set(facecolor=COLOR_SCANNER,  alpha=0.6)
    bp['boxes'][1].set(facecolor=COLOR_PIPELINE, alpha=0.6)

    rng = np.random.default_rng(42)
    sc_s = ax.scatter(rng.normal(1, 0.04, len(s_vals)), s_vals,
                      alpha=0.8, s=60, color='#005F8A',
                      edgecolors='black', linewidth=0.5, zorder=4,
                      label='One recording session')
    ax.scatter(rng.normal(2, 0.04, len(p_vals_clean)), p_vals_clean,
               alpha=0.8, s=60, color="#666699",
               edgecolors='black', linewidth=0.5, zorder=4)

    for i, data in enumerate([s_vals, p_vals_clean], 1):
        ax.annotate(f'{np.mean(data):.1f} mm', xy=(i, np.mean(data)),
                    xytext=(50, 0), textcoords='offset points',
                    fontsize=12, fontweight='bold', va='center')

    ax.legend(handles=[sc_s], loc='upper right', fontsize=9, framealpha=0.8)
    ax.set_ylabel("Mean deviation from own method's mean (mm)", fontsize=12)
    ax.set_title('Intra-Method Repeatability\n(Per-Recording Average)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _legend_lines(ax, y_positions=(0.96, 0.91), x_left=0.82, x_right=0.87, x_text=0.88)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'repeatability_recording_boxplot.png')
    plt.close()


# ============================================================
# BOXPLOT — ELECTRODE-LEVEL REPEATABILITY  (y capped at 10 mm)
# ============================================================

def plot_per_electrode_repeatability_boxplot(
        s_per_elec, p_per_elec_clean, poor_names, output_dir):
    """Y-axis hard-capped at 10 mm. Each dot = one electrode × recording observation."""
    all_scanner_flat = [d for dists in s_per_elec.values()       for d in dists]
    all_pipe_flat    = [d for dists in p_per_elec_clean.values() for d in dists]

    fig, ax = plt.subplots(figsize=(6, 6))

    bp = ax.boxplot(
        [all_scanner_flat, all_pipe_flat],
        tick_labels=['Digitizer\n(Ground Truth)', 'Smartphone Pipeline'],
        patch_artist=True, showmeans=True, meanline=True,
        showfliers=False, widths=0.5,
        medianprops=dict(color='red',  linewidth=2),
        meanprops  =dict(color='blue', linewidth=2, linestyle='--'),
        whiskerprops=dict(linewidth=1.5),
        capprops    =dict(linewidth=1.5))
    bp['boxes'][0].set(facecolor=COLOR_SCANNER,  alpha=0.6)
    bp['boxes'][1].set(facecolor=COLOR_PIPELINE, alpha=0.6)

    rng = np.random.default_rng(42)
    sc_s = ax.scatter(rng.normal(1, 0.04, len(all_scanner_flat)), all_scanner_flat,
                      alpha=0.25, s=15, color='#005F8A', zorder=3,
                      label='One electrode observation')
    ax.scatter(rng.normal(2, 0.04, len(all_pipe_flat)), all_pipe_flat,
               alpha=0.25, s=15, color='#666699', zorder=3)

    for i, data in enumerate([all_scanner_flat, all_pipe_flat], 1):
        ax.annotate(f'{np.mean(data):.1f} mm', xy=(i, np.mean(data)),
                    xytext=(50, 0), textcoords='offset points',
                    fontsize=12, fontweight='bold', va='center')

    ax.set_ylim(0, 10)
    ax.legend(handles=[sc_s], loc='upper right', fontsize=9, framealpha=0.8)
    ax.set_ylabel("Mean deviation from own method's mean (mm)", fontsize=12)
    ax.set_title('Intra-Method Repeatability\n(Per-Electrode Distribution)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    _legend_lines(ax, y_positions=(0.96, 0.91), x_left=0.82, x_right=0.87, x_text=0.88)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'repeatability_electrode_boxplot.png')
    plt.close()


# ============================================================
# HEAD MAPS
# ============================================================

def plot_accuracy_headmap(errors, positions, output_dir, title, subtitle,
                          filename, vmax=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.RdYlGn_r
    _draw_head(ax)
    _plot_electrodes_on_ax(ax, errors, positions, cmap, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label('Mean Error (mm)', fontsize=12)
    ax.text(0,  125, 'Front (NAS)', ha='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(0, -115, 'Back',        ha='center', fontsize=12, fontweight='bold', color='gray')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=55)
    ax.text(0.5, 1.08, subtitle,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, style='italic', color='#555555')
    _save_plot(fig, output_dir, filename)
    plt.close()


def plot_side_by_side_head_maps(errors_left, errors_right, positions, output_dir,
                                title_main, subtitle, title_left, title_right,
                                filename, vmax=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    cmap = plt.cm.RdYlGn_r
    fig.suptitle(title_main, fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.92, subtitle, ha='center', fontsize=14,
             style='italic', color='#555555')

    for ax, errors, title in [(ax1, errors_left, title_left),
                               (ax2, errors_right, title_right)]:
        _draw_head(ax)
        _plot_electrodes_on_ax(ax, errors, positions, cmap, vmax)
        ax.text(0,  125, 'Front (NAS)', ha='center', fontsize=11,
                fontweight='bold', color='gray')
        ax.text(0, -115, 'Back',        ha='center', fontsize=11,
                fontweight='bold', color='gray')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    ax1.text(0, -135, f"Mean Variance: {np.mean(list(errors_left.values())):.1f} mm",
             ha='center', fontsize=14, fontweight='bold', color='#333333')
    ax2.text(0, -135, f"Mean Variance: {np.mean(list(errors_right.values())):.1f} mm",
             ha='center', fontsize=14, fontweight='bold', color='#333333')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02, location='right')
    cbar.set_label('Mean deviation from mean (mm)', fontsize=14)

    _save_plot(fig, output_dir, filename)
    plt.close()


# ============================================================
# STATISTICAL TESTS
# ============================================================

def _extract_icc2(icc_res):
    """
    Pull ICC value and 95% CI from a pingouin intraclass_corr result,
    handling both 'CI95%' (older pingouin) and 'CI95' (newer pingouin)
    column names robustly.
    """
    row = icc_res[icc_res['Type'] == 'ICC2'].iloc[0]
    icc_val = float(row['ICC'])
    # Find whichever CI column exists
    ci_col = next((c for c in icc_res.columns if 'CI' in c.upper()), None)
    ci95   = row[ci_col] if ci_col else 'N/A'
    return icc_val, ci95


def compute_within_method_icc(positions_list, names, method_name="Method"):
    """
    ICC(2,1) – absolute agreement across repeated sessions WITHIN one method.
    Each electrode × axis is a 'Target'; each session/recording is a 'Rater'.
    Requires electrodes present in ALL sessions (common set).
    """
    elec_sets = [set(d.keys()) - set(LANDMARKS) for d in positions_list]
    if not elec_sets or len(positions_list) < 2:
        return None
    common = set.intersection(*elec_sets)
    if not common:
        print(f"  [ICC within {method_name}] No electrode common to all sessions.")
        return None

    rows = []
    for label in sorted(common):
        for sess_name, pos_dict in zip(names, positions_list):
            for ax_i, axis in enumerate(['X', 'Y', 'Z']):
                rows.append({'Target': f"{label}_{axis}",
                             'Rater':  sess_name,
                             'Coord':  pos_dict[label][ax_i]})
    df = pd.DataFrame(rows)
    try:
        icc_res         = pg.intraclass_corr(data=df, targets='Target',
                                             raters='Rater', ratings='Coord')
        icc_val, ci95   = _extract_icc2(icc_res)
        return icc_val, ci95, len(common), len(positions_list)
    except Exception as e:
        print(f"  [ICC within {method_name}] Failed: {e}")
        return None


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
#
# Q1  Is the pipeline ACCURATE?   → Paired Wilcoxon + descriptives
# Q2  Is the pipeline CONSISTENT? → Paired Wilcoxon + ICC + descriptives
# ============================================================


def _extract_icc(icc_result, icc_type='ICC2'):
    """Extract ICC value and 95 % CI (works across pingouin versions)."""
    row = icc_result.set_index('Type').loc[icc_type]
    icc_val = row['ICC']
    ci = None
    for col in ['CI95%', 'CI95']:
        if col in row.index:
            ci = row[col]; break
    if ci is None:
        for col in row.index:
            if 'CI' in str(col):
                ci = row[col]; break
    if ci is not None and hasattr(ci, '__len__') and len(ci) == 2:
        return icc_val, ci[0], ci[1]
    return icc_val, np.nan, np.nan


def _desc(values, label):
    """One-line descriptive block."""
    a = np.asarray(values)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    print(f"    {label}")
    print(f"      n = {len(a)},  Mean = {np.mean(a):.2f} mm,  "
          f"Median = {med:.2f} mm,  SD = {np.std(a, ddof=1):.2f} mm")


def _build_within_icc_df(session_dicts, session_names, electrode_labels):
    """Long-format DataFrame for within-method ICC."""
    common = set(electrode_labels)
    for d in session_dicts:
        common &= set(d.keys())
    common -= set(LANDMARKS)
    common = sorted(common)
    rows = []
    for name, d in zip(session_names, session_dicts):
        for label in common:
            for i, axis in enumerate(['X', 'Y', 'Z']):
                rows.append({'Target': f"{label}_{axis}",
                             'Session': name,
                             'Coord':   d[label][i]})
    return pd.DataFrame(rows), common


# ============================================================

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
#
# Q1  Is the pipeline ACCURATE?   → Paired Wilcoxon + ICC between + descriptives
# Q2  Is the pipeline CONSISTENT? → Paired Wilcoxon + ICC within  + descriptives
# ============================================================


def _extract_icc(icc_result, icc_type='ICC2'):
    """Extract ICC value and 95 % CI (works across pingouin versions)."""
    row = icc_result.set_index('Type').loc[icc_type]
    icc_val = row['ICC']
    ci = None
    for col in ['CI95%', 'CI95']:
        if col in row.index:
            ci = row[col]; break
    if ci is None:
        for col in row.index:
            if 'CI' in str(col):
                ci = row[col]; break
    if ci is not None and hasattr(ci, '__len__') and len(ci) == 2:
        return icc_val, ci[0], ci[1]
    return icc_val, np.nan, np.nan


def _desc(values, label):
    """One-line descriptive block."""
    a = np.asarray(values)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    print(f"    {label}")
    print(f"      n = {len(a)},  Mean = {np.mean(a):.2f} mm,  "
          f"Median = {med:.2f} mm,  SD = {np.std(a, ddof=1):.2f} mm")

def _build_within_icc_df(session_dicts, session_names, electrode_labels):
    """Long-format DataFrame for within-method ICC."""
    common = set(electrode_labels)
    for d in session_dicts:
        common &= set(d.keys())
    common -= set(LANDMARKS)
    common = sorted(common)
    rows = []
    for name, d in zip(session_names, session_dicts):
        for label in common:
            for i, axis in enumerate(['X', 'Y', 'Z']):
                rows.append({'Target': f"{label}_{axis}",
                             'Session': name,
                             'Coord':   d[label][i]})
    return pd.DataFrame(rows), common


# ============================================================

def run_statistical_tests(
    acc_errors,                        # {electrode: euclid_dist}
    s_mean, p_mean_clean, common_clean,
    s_dicts, s_names,                  # raw scanner session dicts
    p_clean, p_clean_names,            # filtered pipeline session dicts
    s_file_errors, p_errors_clean,     # per-recording mean residuals
    s_dist, p_dist_clean, poor_names,  # per-recording per-electrode dists
    s_var, p_var_clean,                # per-electrode variability dicts
):
    common_clean = sorted(common_clean)

    # --- build per-electrode mean variability for each method ---
    s_per_elec = {k: v['mean_distance'] for k, v in s_var.items()
                  if k not in LANDMARKS}
    p_per_elec = {k: v['mean_distance'] for k, v in p_var_clean.items()
                  if k not in LANDMARKS}

    print("\n" + "=" * 65)
    print("  STATISTICAL ANALYSIS")
    print("=" * 65)

    acc_vals = np.array([acc_errors[e] for e in common_clean])

    # ==============================================================
    #  Q1.  IS THE PIPELINE ACCURATE?
    # ==============================================================
    print("\n" + "=" * 65)
    print("  Q1. IS THE PIPELINE ACCURATE?")
    print("=" * 65)
    print("  Logic: compare each electrode's pipeline-vs-digitizer error")
    print("  against the digitizer's OWN noise for that electrode.")
    print("  If no significant difference → pipeline error ≈ digitizer noise.\n")

    # --- Descriptives ---
    print("  Descriptives — Inter-method accuracy (Euclidean)")
    _desc(acc_vals, f"Pipeline vs. Digitizer (n={len(acc_vals)} electrodes)")

    # --- Paired Wilcoxon ---
    common_paired = sorted(set(acc_errors) & set(s_per_elec))
    acc_paired  = np.array([acc_errors[e]  for e in common_paired])
    noise_paired = np.array([s_per_elec[e] for e in common_paired])

    print(f"\n  Paired comparison (n={len(common_paired)} electrodes):")
    print(f"    Pipeline accuracy error : {np.mean(acc_paired):.2f} "
          f"± {np.std(acc_paired, ddof=1):.2f} mm")
    print(f"    Digitizer noise floor   : {np.mean(noise_paired):.2f} "
          f"± {np.std(noise_paired, ddof=1):.2f} mm")

    W, p = stats.wilcoxon(acc_paired, noise_paired)
    print(f"    Wilcoxon signed-rank: W = {W:.1f}, p = {p:.4f}")
    if p > 0.05:
        print("    ✓ No significant difference — pipeline accuracy is "
              "within the digitizer's own measurement noise.")
    else:
        print("    ✗ Pipeline accuracy error significantly exceeds "
              "the digitizer's noise floor.")

    # --- ICC between methods (agreement of mean positions) ---
    print(f"\n  ICC — Between-method agreement")
    print("  " + "-" * 60)
    icc_rows = []
    for label in common_clean:
        for i, axis in enumerate(['X', 'Y', 'Z']):
            icc_rows.append({'Target': f"{label}_{axis}",
                             'Method': 'Digitizer',
                             'Coord':  s_mean[label][i]})
            icc_rows.append({'Target': f"{label}_{axis}",
                             'Method': 'Pipeline',
                             'Coord':  p_mean_clean[label][i]})
    icc_df = pd.DataFrame(icc_rows)
    try:
        icc_res = pg.intraclass_corr(data=icc_df, targets='Target',
                                      raters='Method', ratings='Coord')
        icc_val, ci_lo, ci_hi = _extract_icc(icc_res, 'ICC2')
        q = ("Excellent" if icc_val > 0.90 else "Good" if icc_val > 0.75
             else "Moderate" if icc_val > 0.50 else "Poor")
        print(f"    ICC(2,1) = {icc_val:.4f}  "
              f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  → {q}")
    except Exception as e:
        print(f"    Between-method ICC failed — {e}")

    # ==============================================================
    #  Q2.  IS THE PIPELINE CONSISTENT?
    # ==============================================================
    print("\n" + "=" * 65)
    print("  Q2. IS THE PIPELINE CONSISTENT?")
    print("=" * 65)
    print("  Logic: compare each electrode's spread (variability) in the")
    print("  pipeline against its spread in the digitizer.")
    print("  If no significant difference → pipeline repeats as well.\n")

    # --- Descriptives ---
    s_rec = np.array(list(s_file_errors.values()))
    p_rec = np.array(list(p_errors_clean.values()))
    print("  Descriptives — Per-recording mean residuals")
    _desc(s_rec, f"Digitizer  (n={len(s_rec)} sessions)")
    _desc(p_rec, f"Pipeline   (n={len(p_rec)} recordings)")

    # --- Paired Wilcoxon per electrode ---
    common_var = sorted(set(s_per_elec) & set(p_per_elec))
    s_var_paired = np.array([s_per_elec[e] for e in common_var])
    p_var_paired = np.array([p_per_elec[e] for e in common_var])

    print(f"\n  Paired comparison (n={len(common_var)} electrodes):")
    print(f"    Digitizer variability : {np.mean(s_var_paired):.2f} "
          f"± {np.std(s_var_paired, ddof=1):.2f} mm")
    print(f"    Pipeline variability  : {np.mean(p_var_paired):.2f} "
          f"± {np.std(p_var_paired, ddof=1):.2f} mm")

    W, p = stats.wilcoxon(s_var_paired, p_var_paired)
    print(f"    Wilcoxon signed-rank: W = {W:.1f}, p = {p:.4f}")
    if p > 0.05:
        print("    ✓ No significant difference — pipeline consistency "
              "matches the digitizer electrode-by-electrode.")
    else:
        print("    ✗ Significant difference in consistency between methods.")

    # --- ICC within each method (supervisor's recommendation) ---
    print(f"\n  ICC — Within-method repeatability")
    print("  " + "-" * 60)
    for label, dicts, names in [("Digitizer", s_dicts, s_names),
                                ("Pipeline",  p_clean, p_clean_names)]:
        if len(names) < 2:
            print(f"    {label}: < 2 sessions, skipped"); continue
        df, elecs = _build_within_icc_df(dicts, names, list(common_clean))
        try:
            res = pg.intraclass_corr(data=df, targets='Target',
                                     raters='Session', ratings='Coord')
            val, lo, hi = _extract_icc(res, 'ICC2')
            q = ("Excellent" if val > 0.90 else "Good" if val > 0.75
                 else "Moderate" if val > 0.50 else "Poor")
            print(f"    {label:10s}  ICC(2,1) = {val:.4f}  "
                  f"95% CI [{lo:.4f}, {hi:.4f}]  → {q}")
        except Exception as e:
            print(f"    {label}: ICC failed — {e}")

    print("\n" + "=" * 65)
    print("  END OF STATISTICAL ANALYSIS")
    print("=" * 65)

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_positions = parse_elp_montage(STANDARD_MONTAGE_PATH)

    # --- SCANNER LOAD ---
    s_paths = sorted(glob(os.path.join(SCANNER_DIR, "*.elc")))
    s_names = [os.path.basename(f) for f in s_paths]
    s_dicts = [parse_scanner_elc(f) for f in s_paths]
    (s_mean, s_var, s_file_errors,
     s_dist, s_per_elec) = compute_mean_positions_with_residuals(s_dicts, s_names)
    if not plot_positions:
        plot_positions = s_mean

    # --- PIPELINE LOAD ---
    p_paths = sorted(glob(os.path.join(RESULTS_DIR, "*/electrodes_3d.json")))
    p_names = [os.path.basename(os.path.dirname(f)) for f in p_paths]
    p_raw   = [parse_pipeline_json(f) for f in p_paths]

    p_matched, p_matched_names = [], []
    for i, d in enumerate(p_raw):
        m = align_and_match(d, s_mean)
        if m:
            p_matched.append(m)
            p_matched_names.append(p_names[i])

    (p_mean_all, p_var_all, p_errors_all,
     p_dist_all, p_per_elec_all) = compute_mean_positions_with_residuals(
        p_matched, p_matched_names)
    p_label_map, _ = create_clean_labels(p_matched_names, "Video")

    # --- FILTER POOR RECONSTRUCTIONS ---
    poor_recon    = find_poor_reconstructions(p_errors_all, p_label_map)
    clean_idx     = [i for i, n in enumerate(p_matched_names) if n not in poor_recon]
    p_clean       = [p_matched[i]       for i in clean_idx]
    p_clean_names = [p_matched_names[i] for i in clean_idx]

    (p_mean_clean, p_var_clean, p_errors_clean,
     p_dist_clean, p_per_elec_clean) = compute_mean_positions_with_residuals(
        p_clean, p_clean_names)

    # Remove landmarks from per-electrode dicts
    s_per_elec_elec      = {k: v for k, v in s_per_elec.items()      if k not in LANDMARKS}
    p_per_elec_clean_elec= {k: v for k, v in p_per_elec_clean.items() if k not in LANDMARKS}

    # --- ACCURACY ---
    common_clean = set(s_mean) & set(p_mean_clean) - set(LANDMARKS)
    acc_clean    = {l: np.linalg.norm(s_mean[l] - p_mean_clean[l]) for l in common_clean}

    # ============================================================
    print("\n=== GENERATING PLOTS ===")

    # 1. Accuracy boxplot (electrode labels on scatter points)
    plot_accuracy_boxplot(acc_clean, OUTPUT_DIR)
    print("  [1/5] accuracy_boxplot")

    # 2. Recording-level repeatability (recording labels on scatter points)
    plot_poster_precision_boxplot(s_file_errors, p_errors_clean, OUTPUT_DIR)
    print("  [2/5] repeatability_recording_boxplot")

    # 3. Electrode-level repeatability (y capped at 10 mm, electrode-colored dots)
    plot_per_electrode_repeatability_boxplot(
        s_per_elec_elec, p_per_elec_clean_elec, poor_recon, OUTPUT_DIR)
    print("  [3/5] repeatability_electrode_boxplot")

    # 4. Accuracy head map
    mean_clean = np.mean(list(acc_clean.values()))
    plot_accuracy_headmap(
        errors=acc_clean, positions=plot_positions, output_dir=OUTPUT_DIR,
        title="Inter-Method Accuracy\n(Pipeline vs. Digitizer Mean)",
        subtitle=f"Mean Euclidean distance to digitizer mean: {mean_clean:.1f} mm",
        filename="accuracy_headmap.png", vmax=10)
    print("  [4/5] accuracy_headmap")

    # 5. Side-by-side variability head maps
    sv       = {k: v['mean_distance'] for k, v in s_var.items()       if k not in LANDMARKS}
    pv_clean = {k: v['mean_distance'] for k, v in p_var_clean.items() if k not in LANDMARKS}
    plot_side_by_side_head_maps(
        errors_left=sv, errors_right=pv_clean, positions=plot_positions,
        output_dir=OUTPUT_DIR,
        title_main="Repeatability Comparison (Spatial Variance)",
        subtitle="Internal consistency: Digitizer vs. Smartphone Pipeline",
        title_left=f"Digitizer (n={len(s_paths)} sessions)",
        title_right=f"Pipeline  (n={len(p_clean_names)} clean recordings)",
        filename="variability_headmap_comparison.png", vmax=10)
    print("  [5/5] variability_headmap_comparison")


    run_statistical_tests(
        acc_errors     = acc_clean,
        s_mean         = s_mean,
        p_mean_clean   = p_mean_clean,
        common_clean   = common_clean,
        s_dicts        = s_dicts,
        s_names        = [os.path.basename(f) for f in s_paths],
        p_clean        = p_clean,
        p_clean_names  = p_clean_names,
        s_file_errors  = s_file_errors,
        p_errors_clean = p_errors_clean,
        s_dist         = s_dist,
        p_dist_clean   = p_dist_clean,
        poor_names     = poor_recon,
        s_var          = s_var,            
        p_var_clean    = p_var_clean,      
    )


if __name__ == "__main__":
    main()