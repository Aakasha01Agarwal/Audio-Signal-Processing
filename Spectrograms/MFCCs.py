import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#  Load and audio file

audio_file = '//PATH OF THE >WAV FILE'
audio, sr = librosa.load(audio_file)

# print(sr)

# EXTRACT MFCCs
mfcc = librosa.feature.mfcc(audio, n_mfcc=13, sr=sr)
print(mfcc.shape)

# Visualize MFCCs
plt.figure()
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2f")

# Calculate first and second derivatives of MFCCs
delta_mfcc = librosa.feature.delta(mfcc)
delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)

# Visualize derivatives of MFCCs
plt.figure()
librosa.display.specshow(delta_mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2f")

plt.figure()
librosa.display.specshow(delta_delta_mfcc, x_axis='time', sr=sr)
plt.colorbar(format="%+2f")

mfcc_all = np.concatenate(mfcc, delta_mfcc, delta_delta_mfcc)

plt.show()
