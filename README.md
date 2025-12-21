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
   EEG electrodes are detected and tracked across video frames using YOLO and SAM2, producing stable 2D electrode trajectories. A head-centered 2D coordinate system is defined using anatomical landmarks.

2. **3D Head Reconstruction (Script 2)**  
   A subject-specific 3D head surface is reconstructed from the same video using VGGT, including depth estimation and camera pose recovery.

3. **2D-to-3D Projection and Head Coordinate Alignment (Script 3)**  
   The tracked 2D electrode positions are projected onto the reconstructed 3D head surface, snapped to the scalp, and transformed into a standardized head coordinate system with metric units (mm).
.


## Pipeline Overview
## Pipeline Workflow

The pipeline is organized into three sequential scripts that transform a short video
into head-aligned 3D EEG electrode coordinates.

```mermaid
flowchart TD
    %% Global Inputs
    Video[("Video Input .mp4")]

    %% ==========================================
    %% SCRIPT 1: TRACKING
    %% ==========================================
    subgraph Script1 [Script 1: Detection & Tracking]
        direction TB
        Crop["Interactive Crop"]
        Extract["Extract Frames"]
        Mask["Cap Masking\n(SAM2 + User)"]
        Landmarks["User Clicks Landmarks\n(NAS, LPA, RPA, INION)"]
        Detect["Electrode Detection\n(YOLOv11)"]
        Track["Video Tracking\n(SAM2)"]
        
        Crop --> Extract
        Extract --> Mask
        Mask --> Landmarks
        Landmarks --> Detect
        Detect --> Track
    end

    %% Data Flow from Script 1
    Video --> Crop
    Extract -.->|Saved Frames| FramesDir["./frames/ folder"]
    Crop -.->|Crop Metadata| CropData["crop_info.json"]
    Track -->|"Raw 2D Coords"| TrackData["tracking_results.pkl"]

    %% ==========================================
    %% SCRIPT 2: RECONSTRUCTION
    %% ==========================================
    subgraph Script2 [Script 2: 3D Reconstruction]
        direction TB
        Load["Load Frames from Script 1"]
        Resize["Resize & Pad to 518px"]
        VGGT["VGGT AI Model\n(Depth & Camera Pose)"]
        Map["Save Frame Mapping"]
        
        Load --> Resize
        Resize --> VGGT
        VGGT --> Map
    end

    %% Data Flow to Script 2
    FramesDir --> Load

    %% ==========================================
    %% SCRIPT 3: BRIDGE
    %% ==========================================
    subgraph Script3 [Script 3: The Bridge]
        direction TB
        Match["Match Frames\n(Using Explicit Mapping)"]
        Transform["Transform 2D Coords\n(Crop Space -> VGGT Space)"]
        Unproject["Unproject to 3D\n(Pixel + Depth -> X,Y,Z)"]
        Avg["Robust 3D Averaging\n(Remove Outliers)"]
        Align["Head Alignment\n(Using Landmarks)"]
        
        Match --> Transform
        Transform --> Unproject
        Unproject --> Avg
        Avg --> Align
    end

    %% Data Flow to Script 3
    TrackData --> Match
    VGGT -.->|"Depth & Intrinsics"| ReconData["reconstruction.npz"]
    ReconData --> Unproject
    CropData --> Transform
    
    %% Final Output
    Align --> FinalOutput[("Final Output\nelectrodes_3d.json")]
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
**Command:** `python scripts/pipeline_step1_tracking.py`

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
* **Goal:** Define the head coordinate system for alignment.
* **Action:** In the main "Pipeline" window, click these **4 points** in exact order:
    1.  **Nose (NAS)**
    2.  **Left Ear (LPA)**
    3.  **Right Ear (RPA)**
    4.  **Inion (Back of Head)**

#### **4. Electrode Detection (The "Sweep & Fill" Strategy)**
* **Goal:** Label all electrodes exactly once.
* **Step A (Main Sweep):** Move to a frame (using `A`/`S`) where the most electrodes are visible (usually the front view). Press `D` to run YOLO Auto-Detection. (Do this only ONCE).
* **Step B (Manual Fill):** Move to side/back views where hidden electrodes appear. Manually **Click** on any new electrodes that were not detected in Step A.
* *Warning: Do NOT re-click electrodes that already have a dot. The tracker knows where they are.*

#### **5. Finish & Track**
* Once all electrodes have a unique ID/dot, press `SPACE`.
* The script will close the GUI and run the SAM2 tracker. Wait for the progress bar to reach 100%.

---

### **Step 2: 3D Reconstruction (VGGT)**
**Command:** `python scripts/pipeline_step2_reconstruction.py`

* **Goal:** Build the 3D geometry of the head using the frames extracted in Step 1.
* **Action:** The script runs automatically.
    * *First run:* It will download the VGGT model weights (4GB).
    * *Runtime:* ~2 minutes on GPU, ~1-2 hours on CPU.
    * *Visualization:* At the end, a 3D viewer will open showing the point cloud. Close it to finish.

---

### **Step 3: The Bridge (Fusion)**
**Command:** `python scripts/run_bridge.py`

* **Goal:** Combine 2D tracking data with 3D depth to calculate final coordinates.
* **Action:** The script runs automatically. It matches frames, unprojects 2D pixels to 3D, removes outliers, and aligns the coordinates to the head landmarks.

---

## Outputs

The results are saved in the `results/` folder:

| File | Description |
| :--- | :--- |
| **`electrodes_3d.json`** | **FINAL OUTPUT.** The 3D (X, Y, Z) coordinates of all electrodes in millimeters, aligned to the head center. |
| `electrodes_3d.ply` | A 3D file of the electrodes you can open in MeshLab or Blender to visualize the positions. |
| `tracking_results.pkl` | Raw 2D tracking data from Step 1 (Frame-by-frame pixel coordinates). |
| `aligned_positions.json` | Averaged 2D positions from Step 1 (Useful for 2D topology plots). |
| `reconstruction.npz` | The dense 3D point cloud and camera parameters generated by VGGT. |

## References

1.  **Clausner, T., Dalal, S. S., & Crespo-García, M. (2017).** Photogrammetry-Based Head Digitization for Rapid and Accurate Localization of EEG Electrodes and MEG Fiducial Markers Using a Single Digital SLR Camera. *Frontiers in Neuroscience*, 11, 264.
2.  **Homölle, S., & Oostenveld, R. (2019).** Using a structured-light 3D scanner to improve EEG source modeling with more accurate electrode positions. *Journal of Neuroscience Methods*, 326, 108378.
3.  **Jocher, G., et al. (2024).** Ultralytics YOLO. Available at: https://github.com/ultralytics/ultralytics.
4.  **Ravi, N., et al. (2024).** SAM 2: Segment Anything in Images and Videos. Available at: https://github.com/facebookresearch/sam2.
5.  **Reis, P. M. R., & Lochmann, M. (2015).** Using a motion capture system for spatial localization of EEG electrodes. *Frontiers in Neuroscience*, 9, 130.
6.  **Shirazi, S. Y., & Huang, H. J. (2019).** More Reliable EEG Electrode Digitizing Methods Can Reduce Source Estimation Uncertainty, but Current Methods Already Accurately Identify Brodmann Areas. *Frontiers in Neuroscience*, 13, 1159.
7.  **Taberna, G. A., Marino, M., Ganzetti, M., & Mantini, D. (2019).** Spatial localization of EEG electrodes using 3D scanning. *Journal of Neural Engineering*, 16, 026020.

## Project Timeline

```mermaid
gantt
    title Practical Project: EEG Electrode Registration Toolbox
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

