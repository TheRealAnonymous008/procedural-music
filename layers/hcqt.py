from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import librosa 

class CQT(Layer):
    def __init__(self, 
            trainable=True, 
            name=None, 
            dtype=None, 
            dynamic=False,
            sr:int = 44100,
            n_octaves: int = 7,
            bins_per_semitone: int = 3,
            hop_length_ms: int = 11
        ):
        """
            Apply Constant Q Transform to a given tensor

            Requires: 
            1D tensor or iterable.

            Where:
            sr := sample rate to be used (defaults to 44100)
            n_octaves := number of octaves.
            bins_per_semitone := number of bins associated with each semitone
            hop_length_ms := hop length to use for CQT in ms. Internally, this gets converted to sample rate.
        """

        self.sr = sr 
        self.n_octaves = n_octaves
        self.bins_per_semitone = bins_per_semitone
        self.hop_length_ms = hop_length_ms

        super().__init__(trainable, name, dtype, dynamic)

    def build(self, input_shape : tf.TensorShape):
        return super().build(input_shape)

    def call(self, inputs : tf.Tensor) -> tf.Tensor:
        tf.debugging.Assert(len(inputs.shape) == 1, inputs, 10)

        super().call(inputs)

        hop_length = int(self.sr * self.hop_length_ms/1000.0)
        hop_length = hop_length - (hop_length % 4)

        cqt_results = librosa.cqt(
            np.array(inputs), 
            sr=self.sr, 
            hop_length=hop_length, 
            n_bins=self.bins_per_semitone * self.n_octaves * 12, 
            bins_per_octave=self.bins_per_semitone * 12)

        return tf.abs(cqt_results)
