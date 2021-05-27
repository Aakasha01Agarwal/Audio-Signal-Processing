'''I have used the song THEN NIGHT WE MET by LORD HURON as the audio file and therefore the variables are named accordingly.'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

night_we_met_file = '//PATH OF THE .WAV FILE '
night_we_met, sr = librosa.load(night_we_met_file)

# Extracting Short Time Fourier Transform

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(night_we_met, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale.shape)

# Calculate the spectrogram
Y_scale = np.abs(S_scale) ** 2


# Visualize the spectrogram

def plot_spectrogram(y, sr, hop_length, y_axis='linear'):
    librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")


# plot_spectrogram(Y_scale, sr, HOP_SIZE)

# LOG AMPLITUDE SCALE
Y_log_scale= librosa.power_to_db(Y_scale)
# plot_spectrogram(Y_log_scale, sr, HOP_SIZE)

# Log Frequency scale
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis='log')



plt.show()
