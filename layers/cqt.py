from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import librosa 
import tensorflow.keras.backend as K

class CQT(Layer):
    def __init__(self, 
            trainable=True, 
            name=None, 
            dtype=None, 
            dynamic=False,
            sr:int = 44100,
            n_octaves: int = 7,
            bins_per_semitone: int = 3,
            hop_length: int= 512
        ):
        """
            Apply Constant Q Transform to a given tensor

            Requires: 
            1D tensor or iterable.

            Where:
            sr := sample rate to be used (defaults to 44100)
            n_octaves := number of octaves.
            bins_per_semitone := number of bins associated with each semitone
            hop_length := hop length to use for CQT in samples. 
        """

        self.sr = sr 
        self.n_octaves = n_octaves
        self.bins_per_semitone = bins_per_semitone
        self.hop_length = hop_length

        super().__init__(trainable, name, dtype, dynamic)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "sr": self.sr,
            "n_octaves": self.n_octaves,
            "bins_per_semitone": self.bins_per_semitone,
            "hop_length": self.hop_length
        })
        return config

    def build(self, input_shape : tf.TensorShape):
        return super().build(input_shape)

    def call(self, inputs : tf.Tensor) -> tf.Tensor:

        cqt_results = librosa.cqt(
            np.array(inputs), 
            sr=self.sr, 
            hop_length=self.hop_length, 
            n_bins=self.bins_per_semitone * self.n_octaves * 12, 
            bins_per_octave=self.bins_per_semitone * 12)

        return tf.abs(cqt_results)