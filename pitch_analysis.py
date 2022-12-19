# TODO: Optimization for speed and polyphonic track support.

import librosa
from librosa import pyin
from midi.convert_to_midi import convert_to_midi


def naive_get_pitches(data, sample_rate):
    return pyin(data, sr=sample_rate, fmin=20, fmax=10000)


if __name__ == '__main__':
    data, sample_rate = librosa.load('data/piano.mp3')

    f0, voiced_flags, voiced_probs = naive_get_pitches(data, sample_rate)
    convert_to_midi(f0, sample_rate)

