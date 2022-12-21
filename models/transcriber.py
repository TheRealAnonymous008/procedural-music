import tensorflow as tf 
from tensorflow.keras import models, layers , Input

from layers.cqt import *
from layers.harmonic_stacking import * 

def Transcriber(
    harmonics: list[float],
    batch_size = 1,
) -> models.Model:
    inputs = Input(shape=(None,), ragged=True)

    # TODO update this to use CQT and harmonic stacking.

    return models.Model(inputs=inputs, outputs={"harmonic": inputs})