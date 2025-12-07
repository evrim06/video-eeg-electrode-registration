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
    Start([Start]) --> LoadModels[Load YOLO & SAM2 Models]
    LoadModels --> LoadVideo[Load Video]
    LoadVideo --> DrawHeadROI[User Draws Head ROI]
    DrawHeadROI --> CropFrames[Crop & Save Frames]
    CropFrames --> DrawCapROI[User Draws Cap ROI]
    DrawCapROI --> LoadFrames[Load Cropped Frames]
    LoadFrames --> ClickLandmarks[User Clicks NAS, LPA, RPA]
    ClickLandmarks --> Interactive{Interactive Detection}
    
    Interactive -- Manual Clicks --> ManualElectrodes[User Clicks Electrodes]
    Interactive -- YOLO Detect --> RunYOLO[Run YOLO on Frame]
    RunYOLO --> RestrictYOLO[Filter YOLO Detections by Cap ROI]
    RestrictYOLO --> SuppressDupes[Suppress Duplicates Globally]
    SuppressDupes --> AssignIDs[Assign Electrode IDs]
    ManualElectrodes --> AssignIDs
    
    AssignIDs --> SaveDetections[Save Detection Results]
    SaveDetections --> Tracking[SAM2 Tracking Across Frames]
    Tracking --> Smoothing[Trajectory Smoothing]
    Smoothing --> Ordering[Order Electrodes by Landmarks]
    Ordering --> SaveResults[Save Tracking & Ordering Results]
    SaveResults --> End([End])
```
