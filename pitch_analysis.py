# Using the pYIN algorithm implemented by the librosa package.

import librosa
from librosa import pyin, amplitude_to_db, stft, display, times_like
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data, sample_rate = librosa.load('data/raw notes/C4.mp3')

    f0, voiced_flags, voiced_probs = pyin(data, fmin=20, fmax=10000)
    times = times_like(f0)

    D = amplitude_to_db(np.abs(stft(data)), ref=np.max)
    fig, ax = plt.subplots()
    img = display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

    plt.show()

