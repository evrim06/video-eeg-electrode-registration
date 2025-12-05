from ultralytics import YOLO

# Load model
# model = YOLO("yolo11n.pt")  <-- Starts from scratch

model = YOLO(r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\scripts\yolo\best_v1.pt")  

# Point to your NEW dataset
# Make sure this points to the version with ALL images (Video 1 + Video 2)
yaml_path = r"C:\Users\zugo4834\Downloads\eeg-electrode-detection.v5i.yolov11\data.yaml"

print(f"Fine-tuning model: {model.ckpt_path}")
print(f"On dataset: {yaml_path}")

# Train again
results = model.train(
    data=yaml_path,
    epochs=50, 
    imgsz=640,
    batch=16,
    plots=True
)