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
            A tensor with dimension of at most 3, either
            (data)
            (tracks, data)
            (batches, tracks, data)

            Where:
            sr := sample rate to be used (defaults to 44100)
            n_octaves := number of octaves.
            bins_per_semitone := number of bins associated with each semitone
            hop_length := hop length to use for CQT in samples. 

            Outputs:
            A tensor that represents the CQT applied to the input tensors of the hsape
            (batch, frequuencies, time_steps, 1)
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

    def call(self, x : tf.Tensor) -> tf.Tensor:
        if (len(x.shape) == 1):
            x = tf.stack([x])
        if (len(x.shape) == 2):
            x = tf.stack([x])

        cqt_results = librosa.cqt(
            x.numpy(),
            sr=self.sr, 
            hop_length=self.hop_length, 
            n_bins=self.bins_per_semitone * self.n_octaves * 12, 
            bins_per_octave=self.bins_per_semitone * 12)

        y = tf.abs(cqt_results)

        y = tf.transpose(y, [0, 2, 3, 1])
        return y