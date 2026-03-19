"""
Convert all .elc files from the Scanner_recordings folder to an Excel table.

Usage:
    python elc_to_table.py
"""

import os
import glob
import pandas as pd

# CONFIGURE THIS
SCANNER_DIR  = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\Scanner_recordings"
OUTPUT_EXCEL = r"C:\Users\zugo4834\Desktop\video-eeg-electrode-registration\scanner_electrodes_table.xlsx"

def parse_elc_file(filepath, measurement_id):
    rows = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_positions = False

    for line in lines:
        line = line.strip()
        
        # Stop looking for electrodes once we hit the Labels or HeadShapePoints section
        if line.startswith("Labels"):
            break
            
        # Start looking for electrodes once we hit the Positions section
        if line.startswith("Positions"):
            in_positions = True
            continue
            
        # If we are in the Positions section and the line contains a colon
        if in_positions and ":" in line:
            # Split the line into the name (left of colon) and coordinates (right of colon)
            parts = line.split(":")
            if len(parts) == 2:
                electrode_name = parts[0].strip()
                coords = parts[1].split()
                
                if len(coords) >= 3:
                    try:
                        x = float(coords[0])
                        y = float(coords[1])
                        z = float(coords[2])
                        
                        rows.append({
                            "measurement_id":       measurement_id,
                            "electrode":            electrode_name,
                            "x_position":           round(x, 4),
                            "y_position":           round(y, 4),
                            "z_position":           round(z, 4),
                            "measurement_duration": "" 
                        })
                    except ValueError:
                        continue
                        
    return rows

def main():
    # Find all .elc files in the scanner directory
    pattern = os.path.join(SCANNER_DIR, "*.elc")
    elc_files = sorted(glob.glob(pattern))

    if not elc_files:
        print(f"No .elc files found in:\n  {SCANNER_DIR}")
        return

    print(f"Found {len(elc_files)} .elc file(s):\n")

    all_rows = []
    for filepath in elc_files:
        filename = os.path.basename(filepath)
        measurement_id = os.path.splitext(filename)[0]  # Uses the filename (without .elc) as the ID
        
        rows = parse_elc_file(filepath, measurement_id)
        all_rows.extend(rows)
        
        print(f"  ✓ {filename}  —  {len(rows)} electrodes found")

    # --- Convert to Pandas DataFrame and write to Excel ---
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_excel(OUTPUT_EXCEL, index=False)
        
        print(f"\nDone! Table saved to:\n  {OUTPUT_EXCEL}")
        print(f"Total rows extracted: {len(all_rows)}")
    else:
        print("\nCould not find any coordinate data. Check the file parsing logic.")

if __name__ == "__main__":
    main()