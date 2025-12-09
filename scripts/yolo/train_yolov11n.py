from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Point to the yaml file in your DOWNLOADS folder
# Use 'r' before the string to handle the backslashes correctly
yaml_path = r"C:\Users\zugo4834\Downloads\video-eeg-electrode-registration\data\data.yaml"

print(f"Training on: {yaml_path}")

# Train
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    plots=True
)