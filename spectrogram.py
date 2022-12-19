import math

import matplotlib.pyplot as plt
import numpy as np

from frequency_analysis import *


def plot_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    plt.plot(np.linspace(duration_start_seconds, duration_end_seconds, len(samples)), samples, alpha=0.5)
    plt.show()


def plot_spectrogram(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    plt.specgram(samples[:, 1], Fs=samples_per_second, alpha=0.5)
    plt.show()


def plot_frequencies(duration_start_seconds, duration_send_seconds, audio_data, samples_per_second):
    N, transform, fft_samples = get_frequencies(0, 1, data, sampling_rate)

    plt.xlim(0, 8000)
    plt.plot(fft_samples, 2.0 / N * np.abs(transform[:N // 2]), alpha=0.5)
    plt.show()


if __name__ == '__main__':
    sampling_rate, data = read_file('data/piano.mp3')
    plot_frequencies(1, 1.5, data, sampling_rate)