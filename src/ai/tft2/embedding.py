import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GenericEmbedding(layers.Layer):
    def __init__(self, input_size, embedding_size):
        super().__init__()

        if input_size == 0:
            self._dense = layers.Dense(embedding_size)
        else:
            self._dense = layers.Embedding(input_size, embedding_size)

    def call(self, inputs):
        return self._dense(inputs)
