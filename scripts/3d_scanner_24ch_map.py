from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
#uv pip install mne
import mne

#Load two .elc files
rec_montage = mne.channels.read_custom_montage("Scanner_recordings/24ch_motor_11-12-2025_10-10 AM.elc") #recording
std_montage = mne.channels.read_custom_montage("Scanner_recordings/24ch_motor.elc") #standard


#Recorded montage 2D
rec_montage.plot()
plt.title("Recorded Montage (2D)")
plt.show()

#Standard montage 2D
std_montage.plot()
plt.title("Standard Montage (2D)")
plt.show()

#Recorded montage 3D
fig = rec_montage.plot(kind="3d", show=False)
fig = fig.gca().view_init(azim=70, elev=15)
plt.title("Recorded Montage (3D)")
plt.show()
#Standard montage 3D
fig = std_montage.plot(kind="3d", show=False)
fig = fig.gca().view_init(azim=70, elev=15) 
plt.title("Standard Montage (3D)")
plt.show()