''' I have used the audio file of the song THE NIGHT WE MET by LORD HURON and hence the variables are named accordingly'''

import numpy as np
import librosa
import librosa.display
import numpy as np
from playsound import playsound
import matplotlib.pyplot as plt

night_we_met_file = '//PATH OF THE >WAV FILE'
night_we_met, sr = librosa.load(night_we_met_file)

sample_duration = 1 / sr
duration = sample_duration * len(night_we_met)

# Extract Zero Crossing USING LIBROSA
FRAME_LENGTH = 1024
HOP_LENGHT = 512

zcr_night_we_met = librosa.feature.zero_crossing_rate(night_we_met, FRAME_LENGTH, HOP_LENGHT)[0]
print(zcr_night_we_met.shape)

# plot the Zero Crossing Rate
frames = range(0, zcr_night_we_met.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGHT)

librosa.display.waveplot(night_we_met, alpha=0.7)
plt.plot(t, zcr_night_we_met, color='r')
plt.title("NIGHT WE MET")


plt.show()
