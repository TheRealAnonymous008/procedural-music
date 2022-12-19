import math

from librosa import amplitude_to_db, stft, display
import matplotlib.pyplot as plt
import numpy as np
from frequency_analysis import *
from midiio import read_file


def plot_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    plt.plot(np.linspace(duration_start_seconds, duration_end_seconds, len(samples)), samples, alpha=0.5)
    plt.show()


def plot_spectrogram(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    D = amplitude_to_db(np.abs(stft(samples)), ref=np.max)
    fig, ax = plt.subplots()
    img = display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Spectrum')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    data, sampling_rate = read_file('data/piano.mp3')
    plot_spectrogram(9, 10, data[0], sampling_rate)