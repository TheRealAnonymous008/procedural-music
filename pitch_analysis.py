# Using the pYIN algorithm implemented by the librosa package.

import librosa
from librosa import pyin, times_like
import matplotlib.pyplot as plt

from frequency_analysis import get_samples
from spectrogram import plot_spectrogram


def naive_get_pitches(data, sample_rate):
    return pyin(data, sr=sample_rate, fmin=20, fmax=10000)


def convert_to_midi(f0):
    print(f0)


if __name__ == '__main__':
    data, sample_rate = librosa.load('data/piano.mp3')
    data = get_samples(95, 115, data, sample_rate)

    f0, voiced_flags, voiced_probs = naive_get_pitches(data, sample_rate)
    convert_to_midi(f0)

    fig, ax = plot_spectrogram(0, 60, data, sample_rate)

    times = times_like(f0)
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

    plt.show()

