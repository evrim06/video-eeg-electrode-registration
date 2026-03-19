"""
Convert all electrodes_3d.json files from IMG_xxxx folders to combined CSV and Excel tables.

Usage:
    python json_to_table.py

Edit RESULTS_DIR below to point to your results folder.
"""

import json
import os
import glob
import pandas as pd

# CONFIGURE THIS
RESULTS_DIR  = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\results"
OUTPUT_CSV   = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\electrodes_table.csv"
OUTPUT_EXCEL = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\electrodes_table.xlsx"

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def extract_rows(data, measurement_id):
    rows = []

    # Landmarks first (NAS, LPA, RPA, INION)
    landmarks = data.get("landmarks", {})
    for landmark_name, landmark_data in landmarks.items():
        x, y, z = landmark_data["position"]
        rows.append({
            "measurement_id":       measurement_id,
            "electrode":            landmark_name,
            "x_position":           round(x, 4),
            "y_position":           round(y, 4),
            "z_position":           round(z, 4),
            "measurement_duration": ""
        })

    # Then electrodes
    electrodes = data.get("electrodes", {})
    for electrode_name, electrode_data in electrodes.items():
        x, y, z = electrode_data["position"]
        rows.append({
            "measurement_id":       measurement_id,
            "electrode":            electrode_name,
            "x_position":           round(x, 4),
            "y_position":           round(y, 4),
            "z_position":           round(z, 4),
            "measurement_duration": ""
        })

    return rows

def main():
    # Find all IMG_xxxx folders containing electrodes_3d.json
    pattern = os.path.join(RESULTS_DIR, "IMG_*", "electrodes_3d.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No electrodes_3d.json files found under:\n  {RESULTS_DIR}")
        print("Check that RESULTS_DIR is correct and folders are named IMG_xxxx.")
        return

    print(f"Found {len(json_files)} file(s):\n")

    all_rows = []
    for filepath in json_files:
        folder_name   = os.path.basename(os.path.dirname(filepath))  # e.g. IMG_3841
        measurement_id = folder_name                                 # use folder name as ID
        data = load_json(filepath)
        rows = extract_rows(data, measurement_id)
        all_rows.extend(rows)
        n_landmarks = len(data.get("landmarks", {}))
        n_electrodes = len(data.get("electrodes", {}))
        print(f"  ✓ {folder_name}  —  {n_landmarks} landmarks + {n_electrodes} electrodes")

    #Convert list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(all_rows)

    # Write to CSV
    df.to_csv(OUTPUT_CSV, index=False)

    #Write to Excel
    df.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\nDone! Tables saved to:")
    print(f"  CSV:   {OUTPUT_CSV}")
    print(f"  Excel: {OUTPUT_EXCEL}")
    print(f"Total rows: {len(all_rows)}")

if __name__ == "__main__":
    main()