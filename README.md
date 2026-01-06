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

The pipeline consists of **three main computational stages**:

1. **2D Electrode Detection and Tracking (Script 1)**  
   EEG electrodes and anatomical landmarks are detected and tracked across video frames using YOLO and SAM2, producing stable 2D electrode trajectories.

2. **3D Head Reconstruction (Script 2)**  
   A subject-specific 3D head surface is reconstructed from the same video using VGGT, including depth estimation and camera pose recovery.

3. **2D-to-3D Projection and Head Coordinate Alignment (Script 3)**  
   The tracked 2D electrode positions are projected onto the reconstructed 3D surface, aligned across frames using Procrustes analysis, and transformed into a standardized head coordinate system with metric units (mm). INION is estimated geometrically in 3D space.


## Pipeline Workflow

The pipeline is organized into three sequential scripts that transform a short video
into head-aligned 3D EEG electrode coordinates.
```mermaid
flowchart TD
    %% Global Inputs
    Video[("Video Input.mp4")]

    %% ==========================================
    %% SCRIPT 1: TRACKING (Simplified - 2D Only)
    %% ==========================================
    subgraph Script1 ["Script 1: Detection & Tracking"]
        direction TB
        Crop["1. Interactive Crop(Define ROI)"]
        Extract["2. Extract Frames(every Nth frame)"]
        Mask["3. Cap Masking(SAM2 + User click)"]
        Landmarks["4. User Clicks 3 Landmarks(NAS, LPA, RPA) ! USE DIFFERENT COLORS!"]
        Detect["5. Electrode Detection(YOLOv11 + Manual)"]
        Track["6. Video Tracking(SAM2 propagation)"]
        
        Crop --> Extract
        Extract --> Mask
        Mask --> Landmarks
        Landmarks --> Detect
        Detect --> Track
    end

    %% Script 1 Outputs
    Video --> Crop
    Extract -.->|"Saved Frames"| FramesDir[("./frames/")]
    Crop -.->|"Crop Metadata"| CropInfo["crop_info.json"]
    Track -.->|"Raw 2D Coords"| TrackPkl["tracking_results.pkl"]

    %% ==========================================
    %% SCRIPT 2: RECONSTRUCTION
    %% ==========================================
    subgraph Script2 ["Script 2: 3D Reconstruction"]
        direction TB
        LoadFrames["1. Load Frames(from Script 1)"]
        SelectFrames["2. Select Subset(evenly spaced, max 20)"]
        Resize["3. Resize & Pad(to 518×518)"]
        VGGTModel["4. VGGT Model(Depth + Camera Poses)"]
        SaveMapping["5. Save Frame Mapping(VGGT idx → Script1 idx)"]
        
        LoadFrames --> SelectFrames
        SelectFrames --> Resize
        Resize --> VGGTModel
        VGGTModel --> SaveMapping
    end

    %% Script 2 Data Flow
    FramesDir --> LoadFrames
    SaveMapping -.->|"3D Data + Mapping"| ReconNpz["reconstruction.npz"]

    %% ==========================================
    %% SCRIPT 3: PROJECTION & ALIGNMENT (All Geometry Here!)
    %% ==========================================
    subgraph Script3 ["Script 3: 2D→3D Projection & Alignment"]
        direction TB
        MatchFrames["1. Match Frames(Script1 idx ↔ VGGT idx)"]
        TransformCoords["2. Transform Coordinates(Crop space → VGGT 518px)"]
        Unproject["3. Unproject ALL Frames to 3D(2D pixel + Depth → 3D point)"]
        VirtualRef["4. Build Virtual Reference(Average landmarks from ALL frames)"]
        Procrustes["5. Procrustes Alignment(Align each frame to reference)"]
        RobustAvg["6. Robust 3D Averaging(Outlier removal)"]
        EstimateInion["7. Estimate INION in 3D(Perpendicular to ear axis)"]
        Measure["8. User Measurement(Calipers/Tape/Circumference)"]
        HeadAlign["9. Head Coordinate System(Origin: ear midpoint, scale to mm)"]
        
        MatchFrames --> TransformCoords
        TransformCoords --> Unproject
        Unproject --> VirtualRef
        VirtualRef --> Procrustes
        Procrustes --> RobustAvg
        RobustAvg --> EstimateInion
        EstimateInion --> Measure
        Measure --> HeadAlign
    end

    %% Script 3 Data Flow
    TrackPkl --> MatchFrames
    CropInfo --> TransformCoords
    ReconNpz --> Unproject
    
    %% Final Outputs
    HeadAlign ==>|"FINAL OUTPUT"| FinalJson[("electrodes_3d.json")]
    HeadAlign -.->|"3D Visualization"| FinalPly["electrodes_3d.ply"]
```

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA support (Recommended for fast processing)
- [uv](https://github.com/astral-sh/uv) (Fast Python package installer)

### Setup

1. Clone the repository:
```bash
git clone [https://github.com/your-username/video-eeg-electrode-registration.git](https://github.com/your-username/video-eeg-electrode-registration.git)
cd video-eeg-electrode-registration
```
2. Create and sync the environment using `uv`:

```bash
uv venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate

uv pip install -r requirements.txt
```

## User Guide

This pipeline is divided into three steps. You must run them in order.

### **Step 1: Detection & Tracking**
**Command:** `python scripts/script1_tracking_detection.py`

#### **1. Cropping (Head Selection)**
* **Goal:** Define the Region of Interest (ROI) to help the AI focus.
* **Action:** A "Crop Preview" window will open.
    * Use `A` (Back) and `S` (Forward) to scrub through the video.
    * **Draw ONE box** that is large enough to contain the head in **every** frame (even when the participant turns).
    * Press `SPACE` to confirm.

#### **2. Cap Masking (Defining the Safe Zone)**
* **Goal:** Prevent the AI from detecting background noise (e.g., buttons on a shirt).
* **Action:** A "Confirm Cap Mask" window appears with a yellow overlay.
    * **Recommended:** Press `m` for Manual Mode, then click the **center** of the EEG cap.
    * If the yellow mask covers the cap correctly, press `y` to accept.

#### **3. Landmark Selection (Critical)**
* **Goal:** Define anatomical reference points for head alignment.
* **Action:** In the main "Pipeline" window, click these **3 points** in exact order:
    1. **NAS (Nasion)** - Bridge of nose, between eyebrows
    2. **LPA (Left Pre-Auricular)** - Left ear tragus point
    3. **RPA (Right Pre-Auricular)** - Right ear tragus point

> ! **IMPORTANT:** Use **DIFFERENT colored stickers** for each landmark!
> - NAS: **Red** sticker
> - LPA: **Blue** sticker
> - RPA: **Green** sticker
>
> SAM2 cannot distinguish identical objects. Using the same color will cause tracking errors.

> **Note:** INION (back of head) is estimated automatically in Script 3 using 3D geometry.

#### **4. Electrode Detection (The "Sweep & Fill" Strategy)**
* **Goal:** Label all electrodes exactly once.
* **Step A (Main Sweep):** Move to a frame (using `A`/`S`) where the most electrodes are visible (usually the front view). Press `D` to run YOLO Auto-Detection. (Do this only ONCE).
* **Step B (Manual Fill):** Move to side/back views where hidden electrodes appear. Manually **Click** on any new electrodes that were not detected in Step A.
* Press `R` to refresh the Reference Map and see all detected points.

> ! **Warning:** Do NOT re-click electrodes that already have a dot. The tracker knows where they are.

#### **5. Finish & Track**
* Once all electrodes have a unique ID/dot, press `SPACE`.
* The script will close the GUI and run the SAM2 tracker. Wait for the progress bar to reach 100%.

---

### **Step 2: 3D Reconstruction (VGGT)**

**Command:** `python scripts/script3_bridge.py`

* **Goal:** Build the 3D geometry of the head using the frames extracted in Step 1.
* **Action:** The script runs automatically.
    * *First run:* It will download the VGGT model weights (~4GB).
    * *Runtime:* ~2-5 minutes on GPU, ~1-2 hours on CPU.
    * *Visualization:* At the end, a 3D viewer will open showing the point cloud. Press `Ctrl+C` in the terminal to close it and finish.

---

### **Step 3: 2D→3D Projection & Alignment**

**Command:** `python scripts/script3_bridge.py`

* **Goal:** Project 2D tracking onto 3D, align frames, and scale to real-world millimeters.

#### What This Script Does:
1. **Projects** each tracked 2D point onto the 3D surface using depth maps
2. **Builds a virtual reference** by averaging landmark positions across ALL frames (no need for all 3 landmarks in a single frame)
3. **Aligns each frame** to the reference using Procrustes analysis
4. **Averages** aligned 3D positions with outlier removal
5. **Estimates INION** geometrically in 3D (perpendicular to ear axis)
6. **Scales** to real millimeters using your measurement

#### Head Measurement Input:
The script will pause and ask for a **Head Measurement** to scale the model correctly:

| Option | Description | How to Measure |
|--------|-------------|----------------|
| **1. Caliper** | Direct ear-to-ear distance | Measure straight line between left and right ear tragus |
| **2. Tape Arc** | Arc over top of head | Measure from left ear → top of head → right ear (auto-converted) |
| **3. Circumference** | Head circumference | Measure around the head above the eyebrows (auto-converted) |
| **4. Default** | Use 150mm | Skip measurement (less accurate) |

* **Input:** Type the number of your choice (e.g., `2`) and then the value in millimeters (e.g., `360`).

---

## Outputs

The results are saved in the `results/` folder:

| File | Description |
|:-----|:------------|
| **`electrodes_3d.json`** | **FINAL OUTPUT.** The 3D (X, Y, Z) coordinates of all electrodes in millimeters, aligned to the head coordinate system. |
| `electrodes_3d.ply` | A 3D point cloud file you can open in MeshLab, CloudCompare, or Blender to visualize electrode positions. |
| `tracking_results.pkl` | Raw 2D tracking data from Script 1 (frame-by-frame pixel coordinates). |
| `crop_info.json` | Crop region metadata from Script 1. |
| `vggt_output/reconstruction.npz` | The 3D depth maps, camera parameters, and frame mapping from Script 2. |

### Output Coordinate System

The final `electrodes_3d.json` uses a head-centered coordinate system:
```
        Z (up)
        ↑
        |
        |      Y (front/NAS)
        |     ↗
        |   ↗
        | ↗
        +------------→ X (right/RPA)
       /
      /
     ↙ (left/LPA)

Origin: Midpoint between LPA and RPA (ears)
X-axis: LPA → RPA (left to right)
Y-axis: INION → NAS (back to front)
Z-axis: Down → Up (perpendicular)
Units:  Millimeters (mm)
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| LPA-RPA distance is very small | Same colored stickers | Use DIFFERENT colored stickers for NAS/LPA/RPA |
| NAS-INION distance > 500mm | Landmark tracking confusion | Re-run Script 1 with different colored stickers |
| "No observations for NAS/LPA/RPA" | Landmark not clicked or lost | Re-run Script 1, ensure landmarks are clicked |
| Script 3 shows 0 frames aligned | Less than 2 landmarks visible per frame | Check tracking quality, re-record video with better angles |

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


## Project Timeline

```mermaid
gantt
    title Practical Project: EEG Electrode Registration Pipeline
    dateFormat YYYY-MM-DD
    Familiarize with the topic:2025-10-15,2025-10-22
    Set up project environment(VSCode,UV,Github):2025-10-22,2025-10-29
    Review scripts and plan:2025-10-29,2025-11-09
    Data collection & coding:2025-11-10,2025-12-31
    Add eeg cap types:2026-01-05,2026-01-21
    Testing & documentation: 2026-01-21,2026-01-28
    Preparing final report:2026-01-07,2026-01-28
    Preparing the poster:2026-01-28,2026-02-27

```

