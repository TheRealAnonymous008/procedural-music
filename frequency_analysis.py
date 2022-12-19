from scipy import fft


def get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    sample_start = duration_start_seconds * samples_per_second
    sample_end = duration_end_seconds * samples_per_second
    return audio_data[int(sample_start): int(sample_end)]


def get_frequencies(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second):
    samples = get_samples(duration_start_seconds, duration_end_seconds, audio_data, samples_per_second)[:, 0]
    transform = fft.fft(samples)
    N = len(samples)
    T = 1.0 / samples_per_second
    fft_samples = fft.fftfreq(N, T)[:N//2]
    return N, transform, fft_samples
