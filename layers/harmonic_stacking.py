from tensorflow.keras.layers import Layer
from typing import List
import tensorflow as tf
import numpy as np

class HarmonicStacking(Layer):
    def __init__(self, 
            harmonics: List[float],
            bins_per_semitone: int = 36,
            trainable=True, 
            name=None, 
            dtype=None, 
            dynamic=False
        ):
        super().__init__(trainable, name, dtype, dynamic)

        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [int(tf.math.round(12.0 * bins_per_semitone * tf.math.log(float(h)) / tf.math.log(2.0))) for h in harmonics]

    
    def build(self, input_shape : tf.TensorShape):
        return super().build(input_shape)

    def call(self, inputs : tf.Tensor) -> tf.Tensor:
        tf.assert_equal(tf.shape(inputs).shape == 4)

        channels = []

        for shift in self.shifts:
            if shift == 0:
                padded = input
            elif shift > 0:
                paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
                padded = tf.pad(input[:, :, shift:, :], paddings)
            elif shift < 0:
                paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
                padded = tf.pad(input[:, :, :shift, :], paddings)
            else:
                raise ValueError
            channels.append(padded)

        input = tf.concat(channels, axis=-1)
        input