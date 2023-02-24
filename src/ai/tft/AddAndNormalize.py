
import tensorflow as tf
import keras.api._v2.keras as krs

class AddAndNormalize(krs.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._add = krs.layers.Add()
        self._norm = krs.layers.LayerNormalization()

    def call(self, inputs):
        add_layer = self._add(inputs)
        norm_layer = self._norm(add_layer)

        return norm_layer
