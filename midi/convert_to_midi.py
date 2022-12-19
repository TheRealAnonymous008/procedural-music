import math

import numpy as np
from librosa import hz_to_midi

from midi.midi_representation import Composition, make_midi
from midi.track import Track


def convert_to_midi(f0 : np.ndarray, sample_rate):
    comp: Composition = Composition("Sample")
    comp.add_track(1, 10)

    track: Track = comp.tracks[0]

    curr_note = None
    delta = 0

    track.add_tempo_event(0, 130)
    for x in f0:
        if not math.isnan(x):
            note = int(hz_to_midi(x) + 36) # TODO: Octaves haven't been implemented. Need to remove the +30 eventually
            if curr_note != note:
                if curr_note is not None:
                    track.add_note_off_event(delta, 1, curr_note)
                curr_note = note
                track.add_note_on_event(delta, 1, curr_note, 120)

        else:
            if curr_note is not None:
                track.add_note_off_event(delta, 1, curr_note)
            curr_note = None

        delta += 2      # TODO: Warning! Magic constant. Figure out why we increment by 2. Hypothesis is because it is bpm / 60

    make_midi(comp)
