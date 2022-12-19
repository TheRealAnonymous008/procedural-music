from midiio import read_file
from spectrogram import *

if __name__ == '__main__':
    NOTES = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']

    plt.figure(figsize=(5, 5))
    for note in NOTES:
        data, sampling_rate = read_file('data/raw notes/' + note + '.mp3')
        N, transform, fft_samples = get_frequencies(0, 1, data, sampling_rate)

        plt.xlim(0, 8000)
        plt.plot(fft_samples, 2.0 / N * np.abs(transform[:N // 2]), alpha=0.5, label=note)

    plt.legend()
    plt.show()
