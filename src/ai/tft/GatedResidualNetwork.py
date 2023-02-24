
import tensorflow as tf
import keras.api._v2.keras as krs
from ai.tft.AddAndNormalize import AddAndNormalize
from ai.tft.GatedLinearUnit import GatedLinearUnit

from ai.tft.LinearUnit import LinearUnit

class GatedResidualNetwork(krs.layers.Layer):
    def __init__(self,
        hidden_layer_size: int, use_time_distributed: bool, additional_context: bool,
        output_size: int=None, dropout_rate=None,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        if output_size is None:
            output_size = hidden_layer_size
            self._skip = krs.layers.Activation('linear')
        else:
            skip = krs.layers.Dense(output_size)
            if use_time_distributed:
                self._skip = krs.layers.TimeDistributed(skip)
            else:
                self._skip = skip

        self._linear1 = LinearUnit(hidden_layer_size, None, use_time_distributed)
        self._linear2 = LinearUnit(hidden_layer_size, None, use_time_distributed)

        if additional_context:
            self._context = LinearUnit(hidden_layer_size, None, use_time_distributed, False)
        
        self._elu = krs.layers.ELU() # krs.layers.Activation('elu')
        self._glu = GatedLinearUnit(output_size, None, use_time_distributed, dropout_rate)
        self._ann = AddAndNormalize()

    def call(self, inputs, context=None):
        skip_layer = self._skip(inputs)

        linear_layer1 = self._linear1(inputs)

        if context is None:
            elu_layer = self._elu(linear_layer1)
        else:
            context_layer = self._context(context) + linear_layer1
            elu_layer = self._elu(context_layer)

        linear_layer2 = self._linear2(elu_layer)

        applied_gate_layer, gate_layer = self._glu(linear_layer2)
        ann_layer = self._ann([skip_layer, applied_gate_layer])

        return ann_layer, gate_layer
