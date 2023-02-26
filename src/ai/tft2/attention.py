import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class InterpretableMultiHead(layers.Layer):
    def __init__(self, heads, units):
        super().__init__()
        self._attention = layers.Attention()

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
