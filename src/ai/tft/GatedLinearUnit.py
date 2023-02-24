
import tensorflow as tf
import keras.api._v2.keras as krs

class GatedLinearUnit(krs.layers.Layer):
    def __init__(self,
        hidden_layer_size: int, activation: str, use_time_distributed: bool, dropout_rate=None,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        if dropout_rate is None:
            self._dropout = krs.layers.Activation('linear')
        else:
            self._dropout = krs.layers.Dropout(dropout_rate)

        dense = krs.layers.Dense(hidden_layer_size, activation=activation)
        gate = krs.layers.Dense(hidden_layer_size, activation='sigmoid')
        if use_time_distributed:
            self._dense = krs.layers.TimeDistributed(dense)
            self._gate = krs.layers.TimeDistributed(gate)
        else:
            self._dense = dense
            self._gate = gate
        
        self._multiply = krs.layers.Multiply()
    
    def call(self, inputs):
        dropout_output = self._dropout(inputs)

        dense_layer = self._dense(dropout_output)
        gate_layer = self._gate(dropout_output)

        multiply_layer = self._multiply([dense_layer, gate_layer])

        return multiply_layer, gate_layer