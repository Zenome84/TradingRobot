
import tensorflow as tf
import keras.api._v2.keras as krs

class EmbeddingUnit(krs.layers.Layer):
    def __init__(self,
        output_size: int, category_size: str, use_time_distributed: bool,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        dense = krs.layers.Embedding(category_size, output_size)
        if use_time_distributed:
            self._dense = krs.layers.TimeDistributed(dense)
        else:
            self._dense = dense
    
    def call(self, inputs):
        dense_layer = self._dense(inputs)
        return dense_layer
