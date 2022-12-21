import numpy as np
from scipy.signal import welch
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    sample_start = duration_start_seconds * samples_per_second
    sample_end = duration_end_seconds * samples_per_second
    return audio_data[int(sample_start): int(sample_end)]


def get_frequencies(audio_data, samples_per_second):
    f, power  = welch(audio_data, 
        fs=samples_per_second, 
        detrend=False,
        nfft=512, 
        scaling="spectrum"
    )

    power = scaler.fit_transform(power.reshape((-1, 1))).reshape(-1)
    return f, power
