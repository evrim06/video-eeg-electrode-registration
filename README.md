# video-eeg-electrode-registration
Electroencephalography (EEG) is a non-invasive technique that can measure the neural activity of the brain with high temporal resolution. EEG signals are recording from the scalp by placing several electrodes. Accurate localization of EEG electrodes is essential for reliable brain activity data analysis. Traditional digitizing methods such as ultrasound, motion capture and structured-light 3D scan are reliable methods but can require expensive equipment or complex setup procedures (Shirazi et al., 2019;Taberna et all., 2019;Reis et al., 2015).
Aim of this project is to create a more accessible alternative by developing video-based electrode registration toolbox by using Segment Anything (SAM2). This Python toolbox will automatically detect EEG electrode locations and verify if they are correctly placed according to chosen EEG cap type.
The expected outcome is user-friendly and easy to install Python toolbox that makes EEG electrode registration simpler.


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

## Pipeline Flowchart

```mermaid
flowchart TD
    Start([Start]) --> Init[Initialize Models: YOLO & SAM2]
    
    subgraph Data_Prep [1. Data Preparation]
    Init --> MultiFrameCrop[Interactive Multi-Frame Crop Selection]
    MultiFrameCrop --> ExtractFrames[Extract & Save Cropped Frames]
    end

    subgraph Masking_Phase [2. Cap Mask Generation]
    ExtractFrames --> LoadFirstFrame[Load First Cropped Frame]
    LoadFirstFrame --> AutoMaskGen{Auto or Manual Mask?}
    
    AutoMaskGen -- Auto --> SAM2_Auto[SAM2 AutoMaskGenerator + YOLO]
    AutoMaskGen -- Manual --> Manual_Click[User Clicks Cap Center]
    
    SAM2_Auto --> ExpandMask[Expand Mask Dilation by 10%]
    Manual_Click --> ExpandMask
    
    ExpandMask --> ConfirmMask{User Confirms?}
    ConfirmMask -- No/Redo --> AutoMaskGen
    ConfirmMask -- Yes --> Precompute[Pre-compute & Cache Cap Mask for ALL Frames]
    end

    subgraph Labeling_Phase [3. Interactive Labeling]
    Precompute --> UserClickLandmarks[User Clicks Landmarks: NAS, LPA, RPA]
    UserClickLandmarks --> InteractiveLoop{Interactive Loop}
    
    InteractiveLoop -- Manual Click --> CheckMask1{Inside Cap Mask?}
    CheckMask1 -- Yes --> AddManual[Add Point to SAM2 State]
    CheckMask1 -- No --> Ignore1[Ignore Click]
    
    InteractiveLoop -- Press 'd' --> RunYOLO[Run YOLO on Current Frame]
    RunYOLO --> CheckMask2{Inside Cap Mask?}
    CheckMask2 -- Yes --> CheckDupes{Global Duplicate?}
    CheckDupes -- No --> AddYOLO[Add Point to SAM2 State]
    CheckMask2 -- No --> Ignore2[Ignore Detection]
    
    AddManual --> UpdateDisp[Update HUD & Flash Points]
    AddYOLO --> UpdateDisp
    UpdateDisp --> InteractiveLoop
    end

    subgraph Processing_Phase [4. Tracking & Analysis]
    InteractiveLoop -- Press Space --> Tracking[SAM2 Propagate Electrodes]
    Tracking --> RawSave[Save Raw Tracking Data]
    RawSave --> Smoothing[Savitzky-Golay Smoothing]
    Smoothing --> Ordering[Head-Relative Electrode Ordering]
    Ordering --> FinalSave[Save Smoothed Data & Order JSON]
    end

    FinalSave --> End([End])
```
