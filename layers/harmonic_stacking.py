from tensorflow.keras.layers import Layer
from typing import List
import tensorflow as tf
import numpy as np

class HarmonicStacking(Layer):
    def __init__(self, 
            harmonics: List[float],
            bins_per_semitone: int = 3,
            trainable=True, 
            name=None, 
            dtype=None, 
            dynamic=False
        ):
        """
            Apply Harmonic stacking

            Requires: 
            A tensor with the dimensions
            (batch, frequuencies, time_steps, 1)

            Where:
            harmonics := a list of the harmonics to use for stacking
            bins_per_semitone := the number of bins per semitone to use

            Outputs:
            A tensor of the shape
            (batch, time_steps, frequencies, harmonics)
        """
        super().__init__(trainable, name, dtype, dynamic)

        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [int(tf.math.round(12.0 * bins_per_semitone * tf.math.log(float(h)) / tf.math.log(2.0))) for h in harmonics]

    
    def build(self, input_shape : tf.TensorShape):
        return super().build(input_shape)

    def call(self, x : tf.Tensor) -> tf.Tensor:
        tf.assert_equal(tf.shape(x).shape,  4)

        channels = []

        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
                padded = tf.pad(x[:, :, shift:, :], paddings)
            elif shift < 0:
                paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
                padded = tf.pad(x[:, :, :shift, :], paddings)
            else:
                raise ValueError
            channels.append(padded)

        x = tf.concat(channels, axis=-1)
        y = tf.transpose(x, [0, 2, 1, 3])
        return y