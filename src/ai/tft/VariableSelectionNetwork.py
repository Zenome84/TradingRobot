
import tensorflow as tf
import keras.api._v2.keras as krs

from ai.tft.GatedResidualNetwork import GatedResidualNetwork

class VariableSelectionNetwork(krs.layers.Layer):
    def __init__(self,
        input_size: int, hidden_layer_size: str, dropout_rate: float,
        use_time_distributed: bool, additional_context: bool,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        if additional_context and use_time_distributed:
            self._expand = krs.layers.Lambda(lambda t: krs.backend.expand_dims(t, axis=1))
        else:
            self._expand = krs.layers.Activation('linear')

        if use_time_distributed:
            flatten_shape = [-1, input_size*hidden_layer_size]
        else:
            flatten_shape = [input_size*hidden_layer_size]
        self._flatten = krs.layers.Reshape(flatten_shape)

        self._grn = GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            use_time_distributed=use_time_distributed,
            additional_context=additional_context,
            output_size=input_size,
            dropout_rate=dropout_rate
        )
        self._softmax = krs.layers.Softmax() # krs.layers.Activation('softmax')
        
        self._grns = [GatedResidualNetwork(
            hidden_layer_size=hidden_layer_size,
            use_time_distributed=use_time_distributed,
            additional_context=False,
            dropout_rate=dropout_rate
        ) for _ in range(input_size)]

        self._multiply = krs.layers.Multiply()
    
    def call(self, inputs, context=None):
        
        flattened_inputs = self._flatten(inputs)
        if context is None:
            gated_selection_layer, _ = self._grn(flattened_inputs)
        else:
            expanded_context = self._expand(context)
            gated_selection_layer, _ = self._grn(flattened_inputs, expanded_context)
        variable_selection_layer = krs.backend.expand_dims(
            self._softmax(gated_selection_layer),
            axis=-1
        )

        gathered_inputs = [
            krs.backend.gather(inputs, [i], axis=-2)
            for i in range(len(self._grns))
        ]
        gated_inputs_layer = krs.backend.concatenate([
            grn(gathered_inputs[i])[0]
            for i, grn in enumerate(self._grns)
        ], axis=-2)

        combined_layer = krs.backend.sum(self._multiply([
            variable_selection_layer,
            gated_inputs_layer
        ]), axis=-2)

        return combined_layer, gated_inputs_layer
