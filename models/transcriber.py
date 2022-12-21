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
        # self.dense = layers.Dense(1)
    
    def call(self, inputs : tf.Tensor):
        x = self.cqt(inputs)
        x = self.harmonic_stack(x)
        # x = self.dense(x)

        return x