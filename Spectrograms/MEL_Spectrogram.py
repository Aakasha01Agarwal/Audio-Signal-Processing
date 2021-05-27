'''I have used the song THE NIGHT WE MET by LORD HURON as the audio file and therefore the variables are name accordingly'''
import librosa
import librosa.display
import matplotlib.pyplot as plt

night_we_met_file = '//PATH OF THE .WAV FILE'
night_we_met, sr = librosa.load(night_we_met_file)

# MEl FILTER BANK
filter_bank = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
plt.figure(1)
librosa.display.specshow(filter_bank, sr=sr, x_axis="linear")
plt.colorbar(format="%+2.f")


# Extracting MEL Spectrogram
mel_spectrogram= librosa.feature.melspectrogram(night_we_met, sr= sr, n_fft=2048, hop_length=512, n_mels= 10)
log_mel_spectrogram= librosa.power_to_db(mel_spectrogram)

plt.figure(2)
librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis ='mel', sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
