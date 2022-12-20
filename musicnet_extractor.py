import numpy as np 
import pandas as pd

import sys
sys.path.insert(0, "../")

import midiio
import frequency_analysis as frqa

MIDI_START = 20
MIDI_END = 127

FREQUENCY_COUNT = 257

# Let us define our input and output data
INPUT_COLUMNS = ["f" + str(i) for i in range(0, FREQUENCY_COUNT)]
LABEL_COLUMNS = [i for i in range(MIDI_START, MIDI_END+1)]

def get_data(wav_path, csv_path):
    # Open the midifile to see what is inside. Also open the accopmanying labels.
    data, sample_rate = midiio.read_file(wav_path)

    music_dataframe = pd.read_csv(csv_path)

    music_dataframe
    WINDOW_SAMPLES = sample_rate * 0.01

    # For the model, construct the training data as follows:
    music_dataframe_copy = music_dataframe.copy()
    training_data = pd.DataFrame(columns = ['start_time'] + LABEL_COLUMNS + INPUT_COLUMNS)

    # Create a linear series where points are WINDOW_SAMPLES apart from each other. 
    xs = np.linspace(0, len(data), int (len(data) / WINDOW_SAMPLES), endpoint=False)

    # Iterate over the music dataframe. Construct a one hot encoded vector for this particular time based on the note value
    # At the given time.

    for x in xs:
        notes_on = music_dataframe_copy.query("start_time <= " + str(int(x))).query("end_time >= " + str(int(x)))
        music_dataframe_copy.drop(notes_on.index, axis='index', inplace=True)
        
        note_vec = [0 for i in range(MIDI_START, MIDI_END + 1)]
        if len(notes_on) != 0:
            for n in notes_on['note']:
                note_vec[n - MIDI_START] = 1
        
        
        f, power = frqa.get_frequencies(data[int(x) : int(x + WINDOW_SAMPLES)], sample_rate)
        training_data.loc[len(training_data.index)] = [x, *note_vec, *np.abs(power)]

    return training_data