import tensorflow as tf 
from tensorflow.keras import models, layers , Input, Model

from layers.cqt import *
from layers.harmonic_stacking import * 

class Transcriber(Model):    
    def __init__(self, 
        harmonics : list[float]        
    ):
        super().__init__()

        self.cqt = CQT()
        self.harmonic_stack = HarmonicStacking(harmonics=harmonics)
        
        self.convolutional_stack = [
            [
                layers.Conv2D(16, (5, 5)),
                layers.BatchNormalization(),
                layers.ReLU()
            ], 
            [
                layers.Conv2D(8, (3, 39)),
                layers.BatchNormalization(),
                layers.ReLU()
            ],
            [
                layers.Conv2D(1, (5, 5), activation=tf.nn.sigmoid)
            ]
        ]

        self.pseudo = layers.Dense(1)

    def call(self, inputs : tf.Tensor):
        x = self.cqt(inputs)
        x = self.harmonic_stack(x)

        for conv in self.convolutional_stack:
            for layer in conv:
                x = layer(x)

        x = self.pseudo(x)
        
        return {
            "yn" : tf.unstack(x, axis=-1)
        }