import math

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import fft
from tempfile import mktemp


def read_file(filename: str):
    mp3_audio = AudioSegment.from_file(filename, format='mp3')
    w_name = mktemp('temp.wav')
    mp3_audio.export(w_name, format='wav')
    return wavfile.read(w_name)


def get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    sample_start = duration_start_seconds * samples_per_second
    sample_end = duration_end_seconds * samples_per_second
    return audio_data[int(sample_start): int(sample_end)]


def plot_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    plt.plot(np.linspace(duration_start_seconds, duration_end_seconds, len(samples)), samples, alpha=0.5)
    plt.show()


def plot_spectrogram(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)
    plt.specgram(samples[:, 1], Fs=samples_per_second, alpha=0.5)
    plt.show()


def get_frequencies(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)[:, 0]
    transform = fft.fft(samples)
    N = len(samples)
    T = 1.0 / samples_per_second
    fft_samples = fft.fftfreq(N, T)[:N//2]
    return N, transform, fft_samples


if __name__ == '__main__':
    print("Hello")