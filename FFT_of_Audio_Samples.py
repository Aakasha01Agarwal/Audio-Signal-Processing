''' I used the song THE NIGHT WE MET by LORD HURON as the audio file and therefore the variables are named accordingly '''

import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd

night_we_met_file = '//PATH OF .WAV FILE'
night_we_met, sr = librosa.load(night_we_met_file)

print(sr)  # sample rate

ft = np.fft.fft(night_we_met)

print(ft.shape)

print(ft[0])

magnitude_ft = np.abs(ft)

plt.plot(magnitude_ft)

freq= np.linspace(0, sr, len(magnitude_ft))

plt.plot(freq, magnitude_ft)
plt.show()
