# Using the pYIN algorithm implemented by the librosa package.

import librosa
from librosa import pyin, times_like
import matplotlib.pyplot as plt
from spectrogram import plot_spectrogram

if __name__ == '__main__':
    data, sample_rate = librosa.load('data/piano.mp3')

    f0, voiced_flags, voiced_probs = pyin(data, fmin=20, fmax=10000)
    times = times_like(f0)
    fig, ax = plot_spectrogram(0, 60, data, sample_rate)
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

    plt.show()

