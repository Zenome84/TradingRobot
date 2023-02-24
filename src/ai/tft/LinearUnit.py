
import tensorflow as tf
import keras.api._v2.keras as krs

class LinearUnit(krs.layers.Layer):
    def __init__(self,
        output_size: int, activation: str, use_time_distributed: bool, use_bias=True,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        dense = krs.layers.Dense(output_size, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            self._dense = krs.layers.TimeDistributed(dense)
        else:
            self._dense = dense
    
    def call(self, inputs):
        dense_layer = self._dense(inputs)
        return dense_layer
