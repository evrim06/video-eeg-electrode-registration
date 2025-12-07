from ultralytics import YOLO

# 1. Load your trained model
# (Make sure this points to your latest best.pt)
model = YOLO(r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\runs\detect\train\weights\best_v2.pt")

# 2. Define the path to your NEW video
video_path = "data/IMG_2763.mp4"  # Update this path

print(f"Processing {video_path}...")

# 3. Run Prediction
# save=True   -> Saves a copy of the video with boxes drawn
# conf=0.25   -> Shows all detections with >25% confidence (adjust as needed)
results = model.predict(
    source=video_path,
    save=True,
    conf=0.25
)

print("\nDone! Check the 'runs/detect/predict' folder for your video.")