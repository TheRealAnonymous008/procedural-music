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


def plot_spectrogram(dudio_data):
    D = amplitude_to_db(np.abs(stft(data)), ref=np.max)
    fig, ax = plt.subplots()
    img = display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Spectrum')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.legend(loc='upper right')
    plt.show()


def plot_frequencies(duration_start_seconds, duration_send_seconds, audio_data, samples_per_second):
    N, transform, fft_samples = get_frequencies(0, 1, data, sampling_rate)

    plt.xlim(0, 8000)
    plt.plot(fft_samples, 2.0 / N * np.abs(transform[:N // 2]), alpha=0.5)
    plt.show()


if __name__ == '__main__':
    data, sampling_rate = read_file('data/piano.mp3')
    plot_spectrogram(data)