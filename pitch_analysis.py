# TODO: Implement the YIN Pitch prediction algorithm
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf

from frequency_analysis import get_samples
from midiio import read_file


def plot_acf(samples, window):
    autocorr = acf(samples, adjusted=True, nlags=window)
    plt.plot(autocorr)
    plt.show()


if __name__ == '__main__':
    sampling_rate, data = read_file('data/piano.mp3')

    samples = get_samples(42, 42.1, data, sampling_rate)[:, 1]
    plot_acf(samples, 1000)