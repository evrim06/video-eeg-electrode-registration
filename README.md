# Video EEG Electrode Registration Pipeline

## Overview
Electroencephalography (EEG) is a non-invasive technique that can measure the neural activity of the brain with high temporal resolution. EEG signals are recording from the scalp by placing several electrodes. Accurate localization of EEG electrodes is essential for reliable brain activity data analysis. Traditional digitizing methods such as ultrasound, motion capture and structured-light 3D scan are reliable methods but can require expensive equipment or complex setup procedures (Clausner et al., 2017;Homölle & Oostenveld, 2019;Reis et al., 2015;Shirazi et al., 2019;Taberna et all., 2019;).

## Why This Pipeline?

The goal of this pipeline is to reduce cost, setup complexity, and participant burden while maintaining reasonable spatial accuracy.

Key design principles:

- No dedicated digitization hardware required  
- Minimal manual interaction  
- Subject-specific head geometry (no template head models)  
- Reproducible and modular processing steps  

---

## Pipeline Architecture


The pipeline consists of **two main scripts**:

| Script | Purpose | Key Operations |
|--------|---------|----------------|
| **Script 1** | Annotation & Tracking | Crop → Extract frames → VGGT reconstruction → Sequential landmark annotation (NAS→LPA→RPA) → Electrode annotation (with YOLO quality filtering) → 3D triangulation → Review → Save |
| **Script 2** | 3D Coordinate Processing | Load 3D points → Verify landmarks → User measurement → Coordinate transform → Scale to mm → Export |
---


## Pipeline Workflow

```mermaid
flowchart TD
    Video[("Video Input<br>IMG_3841.mp4")]

    subgraph Script1 ["Script 1: Annotation & Tracking"]
        direction TB
        Select["1. Select Video"]
        Crop["2. Interactive Crop(Define head ROI)"]
        Extract["3. Extract Frames(every Nth frame)"]
        VGGT["4. VGGT Reconstruction(Camera poses + Depth maps)"]
        Landmarks["5. Annotate Landmarks(NAS → LPA → RPA)"]
        Electrodes["6. Annotate Electrodes(Manual + YOLO 80% filter)"]
        Triangulate["7. 3D Triangulation(Multi-view → 3D positions)"]
        Project["8. Project & Track(3D → 2D all frames)"]
        Review["9. Review Tracking(Accept or Redo)"]
        Save["10. Save Results"]
        
        Select --> Crop
        Crop --> Extract
        Extract --> VGGT
        VGGT --> Landmarks
        Landmarks --> Electrodes
        Electrodes --> Triangulate
        Triangulate --> Project
        Project --> Review
        Review -->|Accept| Save
        Review -->|Redo| Landmarks
    end

    Video --> Select
    Save -.->|"Video-specific folder"| ResultsFolder[("results/IMG_3841/")]
    ResultsFolder -.-> TrackPkl["tracking_results.pkl"]
    ResultsFolder -.-> Points3D["points_3d_intermediate.pkl"]
    ResultsFolder -.-> CropInfo["crop_info.json"]
    ResultsFolder -.-> VGGTDir["vggt_output/"]

    subgraph Script2 ["Script 2: 3D Coordinate Processing"]
        direction TB
        SelectRes["1. Select Results Folder"]
        Load3D["2. Load 3D Points"]
        Verify["3. Verify Landmarks"]
        Measure["4. User Measurement"]
        HeadCoord["5. Build Head Coordinate System(includes INION estimation)"]
        Scale["6. Transform to mm"]
        Export["7. Export Files"]
        
        SelectRes --> Load3D
        Load3D --> Verify
        Verify --> Measure
        Measure --> HeadCoord
        HeadCoord --> Scale
        Scale --> Export
    end

    ResultsFolder --> SelectRes
    
    Export ==>|"FINAL OUTPUT"| FinalFolder[("results/IMG_3841/")]
    FinalFolder ==> FinalJson["electrodes_3d.json"]
    FinalFolder -.-> FinalPly["electrodes_3d.ply"]
    FinalFolder -.-> FinalElc["electrodes.elc"]
    
    style FinalJson fill:#90EE90
```
---

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA support (Recommended for fast processing, but CPU works too)
- [uv](https://github.com/astral-sh/uv) (Fast Python package installer)


## User Guide

### Recording Tips (Before You Start)

**For best results:**

1. **Use different colored stickers for landmarks:**
   - NAS (Nasion): RED sticker
   - LPA (Left ear): BLUE sticker  
   - RPA (Right ear): GREEN sticker
2. **Video recording:**
   - Walk slowly around the subject's head (360° coverage)
   - Record for 30-60 seconds
   - Good lighting (avoid shadows)
   - Keep camera 50 cm from head

3. **File organization:**
   - Place videos in `data/Video_Recordings/`
   - Use descriptive names (e.g., `Subject1_Baseline.mp4`)

---

## **Step 1: Annotation & Tracking**

**Command:** `python scripts/script1.py`

#### 1.1 Video Selection
- Lists available videos from `data/Video_Recordings/`
- Creates video-specific results folder (e.g., `results/IMG_3841/`)
- All outputs for this video stored separately

---

#### 1.2 Interactive Cropping
**Goal:** Define region containing the head in ALL frames

**Controls:**
- `A` / `D` - Navigate backward/forward through frames
- Mouse - Draw bounding box around head
- `SPACE` - Confirm and continue

**Tips:**
- Make box **large enough** to contain head in all frames
- Try to keep as much as background information for VGGT reconstruction

---

#### 1.3 VGGT 3D Reconstruction
**Automatic step** - runs after cropping

**Processing:**
- Uses 28 frames evenly distributed across the video
- Runs in isolated subprocess for memory management

**Timing:**
- GPU (NVIDIA): 2-5 minutes
- CPU: 30-60 minutes (be patient!)

> **Note:** On CPU, VGGT will take significantly longer but will complete. There is no timeout - the process runs until finished.

---

#### 1.4 Landmark Annotation (Sequential Windows)

**The script opens 3 separate windows, one for each landmark:**

**Window 1: NAS (Nasion)**
1. Window opens showing "Annotate NAS"
2. Navigate frames with `A` / `D` to find clear views of NAS
3. Click on the NAS landmark (red sticker) in 3-5 different frames
4. Press `Y` to accept
5. Window closes → LPA window opens

**Window 2: LPA (Left Ear)**
1. Window opens showing "Annotate LPA"
2. Navigate to frames where LPA is visible
3. Click on LPA (blue sticker) in 3-5 different frames
4. Press `Y` to accept
5. Window closes → RPA window opens

**Window 3: RPA (Right Ear)**
1. Window opens showing "Annotate RPA"
2. Navigate to frames where RPA is visible
3. Click on RPA (green sticker) in 3-5 different frames
4. Press `Y` to accept
5. Window closes → Electrode annotation window opens

**Controls (same for all landmark windows):**
- `A` / `D` - Navigate frames
- `Left Click` - Add observation for current landmark
- `R` - Clear all annotations for this landmark
- `Y` - Accept and move to next landmark

**Tips:**
- Click each landmark in **3-5 different frames** from different viewing angles
- More observations = better 3D triangulation accuracy
- Use **different colored stickers** for easier identification

**CRITICAL:** Use **different colored stickers** for each landmark!
- NAS: RED sticker
- LPA: BLUE sticker
- RPA: GREEN sticker

---

#### 1.5 Electrode Annotation Window

**After RPA window closes, the electrode annotation window opens.**

**Goal:** Localize all electrodes using manual clicks + YOLO auto-detection

**Controls:**
- `A` / `D` - Navigate frames
- `Left Click` - Add electrode (new or add observation to existing)
- `SPACE` - Run YOLO auto-detection in current frame
- `Q` - Finish annotation (window closes)

**Smart Electrode Matching:**
- When you click on an electrode from a different viewpoint, the script automatically matches it to existing electrodes using 3D position comparison
- This prevents duplicate electrodes when annotating the same electrode from different camera angles
- Console shows "3D match" or "2D match" when adding observations to existing electrodes

**YOLO Quality Filtering:**
- YOLO detections below **80% confidence** are automatically rejected
- This filters out false positives (hair, shadows, reflections)
- Manual clicks are always accepted (no confidence threshold)
- You can adjust the threshold by editing `CONFIG["yolo_conf_accept"]` in script1.py

**Recommended workflow:**

```
1. Navigate to BACK view of head
   → Press SPACE (YOLO auto-detects visible electrodes)
   → Message: "Auto-detected 18 electrodes (rejected 6 low-confidence <80%)"

2. Navigate to LEFT side view
   → Manually click any missing electrodes
   → Click near existing (projected) electrodes to add observations
   → Console shows: "Added observation to E5 (3D match, dist=0.082)"

3. Navigate to RIGHT side view
   → Click remaining missing electrodes
   → Add observations to existing electrodes

4. Navigate to TOP view (if available)
   → Click central electrodes (Cz, Fz, etc.)

5. Press Q when all ~20-24 electrodes are marked
   → Window closes

```
**Tips:** Manual clicks works better than YOLO detections. Try to click as much as electrode in different frames. Click electrodes when camera faces them directly.

**Understanding the interface:**

| Visual Element | Meaning |
|----------------|---------|
| Solid colored circles | Electrodes annotated in current frame |
| Outlined circles with `*` | Projected electrodes from 3D (visible but not annotated in this frame) |
| Header info | Frame number, electrode count |

**YOLO Auto-Detection:**
- Press `SPACE` to run YOLO in current frame
- Detections below 80% confidence are automatically rejected (filters false positives)
- Manual clicks are always accepted (no confidence threshold)
- Use SPACE for bulk detection, then manually add missing electrodes

**Tips:**
- Use `SPACE` for quick bulk detection, then fill gaps manually
- Click each electrode in 3-5 different frames for best accuracy
- Clicking near a projected electrode adds an observation to that electrode
- Don't worry if electrodes only appear in some frames - this is normal due to viewing angles

---

#### 1.6 Review Window

**After pressing `Q` in electrode annotation, the Review window opens.**

**Goal:** Verify that tracking looks correct before saving

**Controls:**
- `A` / `D` - Navigate frames (single step)
- `W` / `S` - Navigate frames (10 frames at a time)
- `P` - Toggle playback (auto-advance through frames)
- `SPACE` - Accept tracking and save results
- `R` - Redo annotation (returns to landmark annotation)
- `Q` - Quit without saving

**What to look for:**
- Electrodes should stay in consistent positions on the head
- Projected points (from 3D) should align with visible electrodes
- Check multiple viewing angles (front, back, sides)

**Interface shows:**
- Frame number and total frames
- PLAY/PAUSE status
- Number of tracked points in current frame
- Whether frame has VGGT data or uses interpolation

**Tips:**
- Use `P` to play through the video and spot any tracking jumps
- If tracking looks wrong, press `R` to redo annotations
- Pay attention to frames marked "interp" (interpolated) - these may be less accurate

---

#### 1.7 Automatic Save

**After pressing `SPACE` in the Review window:**

1. Window closes
2. Script automatically processes and saves results
3. You'll see output in the terminal:

```
=== SAVING RESULTS ===
results/IMG_3841/tracking_results.pkl
results/IMG_3841/points_3d_intermediate.pkl

  Landmarks: 3
  Electrodes: 24
  Total 3D points: 27

============================================================
SCRIPT 1 COMPLETE!
Results saved to: results/IMG_3841
Next: python scripts/script2.py
============================================================
```

**What gets saved:**
- 3D positions for all landmarks and electrodes
- VGGT camera poses and reconstruction
- Frame-by-frame tracking data

**Ready for Script 2!**

---

### **Step 2: 3D Coordinate Processing**

**Command:** `python scripts/script2.py`

#### 2.1 Select Results to Process
- Lists all completed videos
- Choose which one to process
- Outputs saved to same folder

---

#### 2.2 Automatic Loading & Validation

The script loads the 3D points and verifies that all required landmarks are present.

---

#### 2.3 Head Measurement Input

```
Choose measurement method:
  [1] Caliper (ear-to-ear distance)  ← Best accuracy
  [2] Tape measure (arc over head)
  [3] Head circumference
  [4] Use default (150mm, less accurate)
```

**Measurement guide:**

| Method | How to Measure | Accuracy |
|--------|----------------|----------|
| **Caliper** | Straight line: left ear tragus → right ear tragus | ★★★ |
| **Tape Arc** | Arc over top: left ear → vertex → right ear | ★★ |
| **Circumference** | Around head above eyebrows | ★ |

---

## Processing Multiple Videos

```bash
# Process each video sequentially
python scripts/script1.py  # Select [1] IMG_3841
python scripts/script2.py  # Select [1] IMG_3841

python scripts/script1.py  # Select [2] IMG_3842
python scripts/script2.py  # Select [2] IMG_3842

# Continue for all videos...
```

**Results organization:**

```
results/
  ├── IMG_3841/
  │   ├── electrodes_3d.json  ← Final output
  │   └── ... (intermediate files)
  ├── IMG_3842/
  └── Subject1_Baseline/
```

**Renaming folders (optional):**
```bash
# Windows: move results\IMG_3841 results\Subject1_Session1
# Linux/Mac: mv results/IMG_3841 results/Subject1_Session1
```

---

## Outputs

### Primary Output

**`electrodes_3d.json`** - 3D coordinates in head-centered coordinate system

```json
{
  "coordinate_system": {
    "origin": "midpoint between LPA and RPA",
    "x_axis": "left to right (LPA -> RPA)",
    "y_axis": "back to front (INION -> NAS)",
    "z_axis": "down to up",
    "units": "mm"
  },
  "landmarks": {
    "NAS": {
      "position": [0.0, 85.2, 12.3]
    }
  },
  "electrodes": {
    "E0": {
      "position": [12.3, 45.6, 78.9]
    }
  }
}
```

### Coordinate System

```
        Z (up)
        ↑
        |      Y (front/NAS)
        |     ↗
        | ↗
        +--------→ X (right/RPA)
       /
     ↙ (left/LPA)

Origin: Midpoint between LPA and RPA (ears)
Units:  Millimeters (mm)
```

### Additional Files

| File | Use |
|------|-----|
| `electrodes_3d.ply` | 3D visualization (MeshLab, CloudCompare) |
| `electrodes.elc` | EEG software (BESA, FieldTrip, EEGLAB) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **VGGT takes very long on CPU** | This is normal (30-60 min). Be patient or use GPU |
| **VGGT out of memory (GPU)** | Reduce `MAX_VGGT_FRAMES` to 20 in script1.py |
| **"LPA-RPA distance too small"** | Use different colored stickers for landmarks |
| **"Missing landmarks"** | Re-run Script 1, click in 3-5 frames |
| **Duplicate electrodes** | Already handled by 3D matching - click same electrode from different views |
| **YOLO rejects all** | Lower `yolo_conf_accept` to 0.60 |
| **Tracking looks wrong in review** | Press `R` to redo annotations |
| **Asymmetric LPA-RPA** | Check landmark annotations; may have swapped LPA/RPA |

---

## References

1. **Clausner, T., Dalal, S. S., & Crespo-García, M. (2017).** Photogrammetry-Based Head Digitization for Rapid and Accurate Localization of EEG Electrodes and MEG Fiducial Markers Using a Single Digital SLR Camera. *Frontiers in Neuroscience*, 11, 264.
2. **Homölle, S., & Oostenveld, R. (2019).** Using a structured-light 3D scanner to improve EEG source modeling with more accurate electrode positions. *Journal of Neuroscience Methods*, 326, 108378.
3. **Jocher, G., et al. (2024).** Ultralytics YOLO. Available at: https://github.com/ultralytics/ultralytics.
4. **Ravi, N., et al. (2024).** SAM 2: Segment Anything in Images and Videos. Available at: https://github.com/facebookresearch/sam2.
5. **Reis, P. M. R., & Lochmann, M. (2015).** Using a motion capture system for spatial localization of EEG electrodes. *Frontiers in Neuroscience*, 9, 130.
6. **Shirazi, S. Y., & Huang, H. J. (2019).** More Reliable EEG Electrode Digitizing Methods Can Reduce Source Estimation Uncertainty, but Current Methods Already Accurately Identify Brodmann Areas. *Frontiers in Neuroscience*, 13, 1159.
7. **Taberna, G. A., Marino, M., Ganzetti, M., & Mantini, D. (2019).** Spatial localization of EEG electrodes using 3D scanning. *Journal of Neural Engineering*, 16, 026020.
8. **Wang, J., et al. (2025).** VGGT: Visual Geometry Grounded Transformer. Available at: https://github.com/facebookresearch/vggt.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{video_eeg_electrode_registration,
  title = {Video-based EEG Electrode Registration Pipeline},
  author = {Evrim Evirgen},
  year = {2025},
  url = {https://github.com/evrim06/video-eeg-electrode-registration}
}
```

---

## Project Timeline

```mermaid
gantt
    title Practical Project: EEG Electrode Registration Pipeline
    dateFormat YYYY-MM-DD
    Familiarize with the topic:2025-10-15,2025-10-22
    Set up project environment(VSCode,UV,Github):2025-10-22,2025-10-29
    Review scripts and plan:2025-10-29,2025-11-09
    Data collection & coding & testing:2025-11-10,2026-01-19
    Documentation: 2026-01-21,2026-01-28
    Preparing the poster:2026-01-19,2026-02-27

```

