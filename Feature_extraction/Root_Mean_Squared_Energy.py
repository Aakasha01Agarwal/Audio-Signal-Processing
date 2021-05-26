''' I have used the audio file of the song THE NIGHT WE MET by LORD HURON so the variables are named accordingly'''

import numpy as np
import librosa
import librosa.display
import numpy as np
from playsound import playsound
import matplotlib.pyplot as plt

night_we_met_file = '//PATH OF THE >WAV FILE'

night_we_met, sr = librosa.load(night_we_met_file)

print(night_we_met)
print(night_we_met.size)

sample_duration = 1 / sr
duration = sample_duration * len(night_we_met)

# Extract RMSE USING LIBROSA
FRAME_LENGTH = 1024
HOP_LENGHT = 512

def rms_using_librosa():

    # Using Librosa built in funciton to calculate the RMSE
    rms_night_we_met = librosa.feature.rms(night_we_met, frame_length=FRAME_LENGTH, hop_length=HOP_LENGHT)[0]
    print(rms_night_we_met.shape)

    # plot the rmse
    frames = range(0, rms_night_we_met.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGHT)

    librosa.display.waveplot(night_we_met, alpha=0.7)
    plt.plot(t, rms_night_we_met, color='r')
    plt.title("NIGHT WE MET")


# RMSE CODE FROM SCRATCH                ---> Root Meam Squared Energy
def rms(signal, frame_length, hop_length):
    rms = []

    for i in range(0, len(signal), hop_length):
        current_rms = np.sqrt(np.sum(signal[i:i + frame_length] ** 2) / frame_length)
        rms.append(current_rms)

    return np.array(rms)


rms1_night_we_met = rms(night_we_met, frame_length=FRAME_LENGTH, hop_length=HOP_LENGHT)

frames = range(0, rms1_night_we_met.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGHT)
librosa.display.waveplot(night_we_met, alpha=0.7)
plt.plot(t, rms1_night_we_met, color='r')

plt.show()
