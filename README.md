# Video EEG Electrode Registration Toolbox

## Overview
Electroencephalography (EEG) is a non-invasive technique that can measure the neural activity of the brain with high temporal resolution. EEG signals are recording from the scalp by placing several electrodes. Accurate localization of EEG electrodes is essential for reliable brain activity data analysis. Traditional digitizing methods such as ultrasound, motion capture and structured-light 3D scan are reliable methods but can require expensive equipment or complex setup procedures (Shirazi et al., 2019;Taberna et all., 2019;Reis et al., 2015).

## Why This Tool?

Traditional EEG digitization methods—such as ultrasound systems, motion capture, and structured-light 3D scanning—can achieve high localization accuracy but suffer from important practical limitations, including **high hardware costs**, **complex experimental setup**, and **limited accessibility** for smaller laboratories (Taberna et al., 2019; Clausner et al., 2017; Homölle & Oostenveld, 2019).

This Python-based toolbox offers a **user-friendly, easy-to-install solution** that simplifies EEG electrode registration using only a standard camera and computer vision. The method combines YOLOv11, a real-time object detection model, for electrode detection and Segment Anything 2 (SAM2), a foundation model for video segmentation and tracking, for robust electrode propagation across frames (Jocher et al., 2024; Ravi et al., 2024).

## Pipeline Overview
The technical logic follows a 4-stage pipeline: **Data Prep** $\rightarrow$ **Masking** $\rightarrow$ **Labeling** $\rightarrow$ **Tracking**.

```mermaid
flowchart TD
    Start([Start]) --> Init[Initialize Models: YOLO & SAM2]

    %% --------------------
    subgraph Data_Prep [1. Data Preparation]
        direction TB
        Init --> MultiFrameCrop[Interactive Multi-Frame Crop]
        MultiFrameCrop --> ExtractFrames[Extract & Save Frames]
    end

    %% --------------------
    subgraph Masking_Phase [2. Cap Mask Generation]
        direction TB
        ExtractFrames --> LoadFirstFrame[Load First Frame]
        LoadFirstFrame --> AutoMaskGen{Auto or Manual?}
        AutoMaskGen -- Auto --> SAM2_Auto[SAM2 AutoMask + YOLO]
        AutoMaskGen -- Manual --> Manual_Click[User Clicks Cap Center]
        SAM2_Auto --> ExpandMask[Expand Mask +10%]
        Manual_Click --> ExpandMask
        ExpandMask --> Precompute[Cache Cap Mask for All Frames]
    end

    %% --------------------
    subgraph Labeling_Phase [3. Interactive Labeling]
        direction TB
        Precompute --> UserClickLandmarks[Click Landmarks: NAS, LPA, RPA]
        UserClickLandmarks --> InteractiveLoop{User Action Loop}
        InteractiveLoop -- Manual Click --> AddManual[Add Point]
        InteractiveLoop -- Press 'D' --> RunYOLO[Run YOLO Detection]
        RunYOLO --> Filter[Filter Duplicates & Outside Mask]
        Filter --> AddYOLO[Add Points to State]
    end

    %% --------------------
    subgraph Processing_Phase [4. Tracking & Analysis]
        direction TB
        InteractiveLoop -- Press Space --> Tracking[SAM2 Propagation]
        Tracking --> RawSave[Save Raw Data]
        RawSave --> Smoothing[Savitzky-Golay Smoothing]
        Smoothing --> Ordering[Head-Relative Ordering]
        Ordering --> FinalSave[Save JSON & PKL]
    end

    FinalSave --> End([End])
```
## Installation

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA support is recommended for faster SAM2 tracking (CPU is supported but slower)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/video-eeg-electrode-registration.git
cd video-eeg-electrode-registration
```
2. Create and sync the environment using `uv`:
```bash
uv venv
uv pip install -r requirements.txt
```

## User Guide (Interactive Pipeline)
### 1. Cropping (Head Selection)

Goal: Define the Region of Interest (ROI) to help the AI focus.

Action: A "Crop Preview" window will open.

Use A (Back) and S (Forward) to scrub through the video.

Draw ONE box that is large enough to contain the head in every frame (even when the participant turns).

Press SPACE to confirm.

### 2. Cap Masking (Defining the Safe Zone)

Goal: Prevent the AI from detecting background noise (e.g., buttons on a shirt).

Action: A "Confirm Cap Mask" window appears with a yellow overlay.

Recommended: Press m for Manual Mode, then click the center of the EEG cap.

If the yellow mask covers the cap correctly, press y to accept.

### 3. Landmark Selection (Critical)

Goal: Define the head coordinate system.

Action: In the main "Pipeline" window, click these 3 points in exact order:

1) Nose (NAS)

2) Left Ear (LPA)

3) Right Ear (RPA)

### 4. Electrode Detection (The "Sweep & Fill" Strategy)

Goal: Label all electrodes exactly once without creating duplicates.

Step A (Main Sweep)

Move to a frame (using A/S) where the most electrodes are visible (usually the front view).

Press D to run YOLO Auto-Detection. (Do this only ONCE).

Red dots will appear and flash for a few seconds.

Step B (Manual Fill)

Move to side/back views where hidden electrodes appear.

Manually Click on any new electrodes that were not detected in Step A.

**Warning: Do NOT re-click electrodes that already have a dot, even if they moved. The tracker knows where they are.**

### 5. Finish & Track

Once all electrodes have a unique ID/dot, press SPACE.

The script will close the GUI and run the SAM2 tracker.

Wait for the progress bar to reach 100%.

## Outputs

The results are saved in the `results/` folder:

| File                   | Format | Description |
|------------------------|--------|-------------|
| `tracking_smoothed.pkl` | Pickle | Final 2D coordinates (X, Y) for every electrode in every frame, smoothed to remove jitter. |
| `electrode_order.json`  | JSON   | A sorted list of electrode IDs ordered spatially (front-to-back, left-to-right). |
| `crop_info.json`        | JSON   | Metadata about the crop coordinates used for this session. |

## References

1.  **Clausner, T., Dalal, S. S., & Crespo-García, M. (2017).** Photogrammetry-Based Head Digitization for Rapid and Accurate Localization of EEG Electrodes and MEG Fiducial Markers Using a Single Digital SLR Camera. [cite_start]*Frontiers in Neuroscience*, 11, 264. 
2.  **Homölle, S., & Oostenveld, R. (2019).** Using a structured-light 3D scanner to improve EEG source modeling with more accurate electrode positions. [cite_start]*Journal of Neuroscience Methods*, 326, 108378. 
3.  **Jocher, G., et al. (2024).** Ultralytics YOLO. Available at: https://github.com/ultralytics/ultralytics.
4.  **Ravi, N., et al. (2024).** SAM 2: Segment Anything in Images and Videos. [cite_start]*arXiv preprint arXiv:2408.00714*. 
5.  **Reis, P. M. R., & Lochmann, M. (2015).** Using a motion capture system for spatial localization of EEG electrodes. [cite_start]*Frontiers in Neuroscience*, 9, 130. 
6.  **Shirazi, S. Y., & Huang, H. J. (2019).** More Reliable EEG Electrode Digitizing Methods Can Reduce Source Estimation Uncertainty, but Current Methods Already Accurately Identify Brodmann Areas. [cite_start]*Frontiers in Neuroscience*, 13, 1159. 
7.  **Taberna, G. A., Marino, M., Ganzetti, M., & Mantini, D. (2019).** Spatial localization of EEG electrodes using 3D scanning. [cite_start]*Journal of Neural Engineering*, 16, 026020.


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

