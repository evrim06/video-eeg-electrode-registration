import pickle
import matplotlib.pyplot as plt

with open("C:\\Users\\zugo4834\\Desktop\\video-eeg-electrode-registration\\results\\tracking_smoothed.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Frames: {len(data)}")

plt.figure(figsize=(8, 8))

for frame, objs in data.items():
    for obj_id, (x, y) in objs.items():
        if obj_id >= 3:  # skip landmarks
            plt.scatter(x, y, s=3)

plt.gca().invert_yaxis()   # important for image coordinates
plt.title("All tracked electrode positions (smoothed)")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.axis("equal")
plt.show()
