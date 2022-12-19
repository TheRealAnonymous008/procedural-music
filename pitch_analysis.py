# Using the pYIN algorithm implemented by the librosa package.

import librosa
from librosa import pyin, times_like, hz_to_midi
import matplotlib.pyplot as plt
import numpy as np
import math

from frequency_analysis import get_samples
from midi.midi_representation import Composition, make_midi
from midi.track import Track
from spectrogram import plot_spectrogram


def naive_get_pitches(data, sample_rate):
    return pyin(data, sr=sample_rate, fmin=20, fmax=10000)


def convert_to_midi(f0 : np.ndarray, sample_rate):
    comp: Composition = Composition("Sample")
    comp.add_track(1, 10)

    track: Track = comp.tracks[0]

    curr_note = None
    delta = 0

    for x in f0:
        if not math.isnan(x):
            note = int(hz_to_midi(x) + 30)
            if curr_note != note:
                if curr_note is not None:
                    track.add_note_off_event(delta, 1, curr_note)
                curr_note = note
                track.add_note_on_event(delta, 1, curr_note, 120)

        else:
            if curr_note is not None:
                track.add_note_off_event(delta, 1, curr_note)
            curr_note = None

        delta += 2      # TODO: Warning! Magic constant. Figure out why we increment by 2

    make_midi(comp)


if __name__ == '__main__':
    data, sample_rate = librosa.load('data/piano.mp3')

    f0, voiced_flags, voiced_probs = naive_get_pitches(data, sample_rate)
    convert_to_midi(f0, sample_rate)

