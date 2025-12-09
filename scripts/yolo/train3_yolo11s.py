from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data=r"C:\Users\zugo4834\Downloads\eeg-electrode-detection.v5i.yolov11\data.yaml",
    epochs=100,
    imgsz=960,
    batch=8,
    plots=True
)
