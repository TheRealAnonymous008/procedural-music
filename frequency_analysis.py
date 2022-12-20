from scipy import fft
from scipy.signal import welch


def get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    sample_start = duration_start_seconds * samples_per_second
    sample_end = duration_end_seconds * samples_per_second
    return audio_data[int(sample_start): int(sample_end)]


def get_frequencies(audio_data, samples_per_second):
    f, s  = welch(audio_data, 
        fs=samples_per_second, 
        detrend=False,
        nfft=512, 
    )

    return f, s
