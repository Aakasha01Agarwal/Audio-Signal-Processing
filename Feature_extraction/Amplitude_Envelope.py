''' I used the song Night We Met by Lord Huron so the variables are named accordingly'''
import numpy as np
import librosa
import librosa.display
import numpy as np
from playsound import playsound
import matplotlib.pyplot as plt

# Load Audio Files

night_we_met_file = '/REPLACE THIS BY THE PATH OF THE .WAV FILE'

# To play the sound uncomment the line below
# playsound(night_we_met_file)

night_we_met, sr = librosa.load(night_we_met_file)

print(night_we_met)
print(night_we_met.size
      )

sample_duration = 1 / sr
duration = sample_duration * len(night_we_met)

# Visualize

# plt.figure(figsize=(15, 17))

librosa.display.waveplot(night_we_met, alpha=0.7)
plt.title("NIGHT WE MET")

# Calculate Amplitude Envelope using overlapping frames
frame_size = 1024
hop_length = 512


def amplitude_envelope(signal, frame_size, hop_length):
    amplitude_envelope = []
    # Calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length):
        current_max = max(signal[i:i + frame_size])
        amplitude_envelope.append(current_max)

    return np.array(amplitude_envelope)


ae_night_we_met = amplitude_envelope(night_we_met, frame_size, hop_length)
print('Number of Frames: ', len(ae_night_we_met))

# Visualize the amplitude Envelope
frames = range(0, ae_night_we_met.size)
t = librosa.frames_to_time(frames, hop_length=hop_length)

plt.plot(t, ae_night_we_met, color='r')

plt.show()
