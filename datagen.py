import pretty_midi as pmidi
from pretty_midi.instrument import Instrument
import pandas as pd
import numpy as np
from typing import List
import tensorflow as tf

def get_piano_roll(
    instrument_notes : List[pmidi.Note], 
    song_duration_seconds : float,
    delta_time_s : float
    ):
    xs = np.linspace(0, song_duration_seconds, int(song_duration_seconds // delta_time_s)) 
    piano_roll = []
    sorted_notes = sorted(instrument_notes, key=lambda note: note.start)

    idx = 0
    for x in xs: 
        # Order is duration and velocity
        slice_roll = [[0, 0] for i in range(0, 128)]

        for i in range(idx, len(sorted_notes)):
            note = sorted_notes[i]
            if note.start > x:
                slice_roll[note.pitch] = [
                    note.duration,
                    note.velocity
                ]
            else: 
                break
        
        idx += len(slice_roll)
        piano_roll.append(slice_roll)
    return np.array(piano_roll)


def add_silence(seq_length_seconds: float, delta_time_s: float=0.05):
    xs = np.linspace(0, seq_length_seconds, int(seq_length_seconds // delta_time_s)) 
    return np.array([[[0, 0] for _ in range(128)] for __ in xs])

def midi_to_notes(file, delta_time_s : float = 0.05):
    pm = pmidi.PrettyMIDI(file)
    data = []
    
    for instrument in pm.instruments:
        instrument : Instrument = instrument 
        notes = get_piano_roll(instrument.notes, pm.get_end_time(), delta_time_s)
        if len(data) == 0:
            data = notes
        else:
            np.concatenate(data, notes)
            
    return np.array(data)

def create_sequences(df : tf.data.Dataset, seq_length: int):
    seq_length +=1 
    windows = df.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def split_labels(seq):
        inputs = seq[:-1]
        labels = seq[-1]
        return inputs, labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def make_tf_dataset(df : np.array, seq_length : int):
    dataset = tf.data.Dataset.from_tensor_slices(df)
    dataset = create_sequences(dataset, seq_length)
    return dataset

def apply_batching(data, batch_size, n_notes, seq_length):
  buffer_size = n_notes- seq_length 
  train_ds = (data
              .shuffle(buffer_size)
              .batch(batch_size, drop_remainder=True)
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))
  return train_ds

def extract_dataset(paths, seq_length_seconds : float, delta_time_s: float= 0.05, batches: int = 150):
    data = []
    for path in paths:
        notes = midi_to_notes(path, delta_time_s=delta_time_s)
        if len(data) == 0:
            data = notes
        else:
            data = np.concatenate([data, notes, add_silence(seq_length_seconds, delta_time_s)], axis = 0) 

    seq_length = int(seq_length_seconds // delta_time_s)
    df = make_tf_dataset(data, seq_length=seq_length)
    df = apply_batching(df, n_notes = len(data),seq_length=seq_length, batch_size=batches)
    return df

if __name__ == "__main__":
    dataset = extract_dataset(["data/Endless-Piano-Carousel-Part-16.mid", "data/Endless-Piano-Carousel-Part-16.mid"],
        1, 0.5
    )
    print(dataset)