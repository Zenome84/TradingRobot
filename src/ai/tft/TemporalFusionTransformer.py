
from typing import Dict
import tensorflow as tf
import keras.api._v2.keras as krs
from ai.tft.AddAndNormalize import AddAndNormalize
from ai.tft.EmbeddingUnit import EmbeddingUnit
from ai.tft.GatedLinearUnit import GatedLinearUnit

from ai.tft.GatedResidualNetwork import GatedResidualNetwork
from ai.tft.InterpretableMultiHead import InterpretableMultiHeadAttention
from ai.tft.LinearUnit import LinearUnit
from ai.tft.VariableSelectionNetwork import VariableSelectionNetwork

class TemporalFusionTransformer(krs.Model):
    def __init__(self, data_specs, quantile_spec, d_model, num_attn_heads, dropout_rate, *args, **kwargs):
        '''
        
        '''
        super().__init__(*args, **kwargs)
        self._data_specs = {
            'inputs': {
                'static': {
                    'day_of_week': (0, 5)
                },
                'observed': {
                    'time_of_day_ts': (0, 0),
                    'volume_ts': (1, 0),
                    'price_ts': (2, 0)
                },
                'forecast': {
                    'time_of_day_ts': (0, 0)
                }
            },
            'targets': {
                'high_ts': (0, 0),
                'low_ts': (1, 0)
            }
        }
        self._quantile_spec = [0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self._embed: Dict[str, Dict[str, krs.layers.Layer]] = {}
        self._vsn: Dict[str, krs.layers.Layer] = {}
        for input_type, inputs in self._data_specs['inputs'].items():
            use_time_distributed = (input_type != 'static')

            self._vsn[input_type] = VariableSelectionNetwork(
                len(self._data_specs['inputs'][input_type]),
                self.d_model,
                self.dropout_rate,
                use_time_distributed,
                use_time_distributed
            )

            self._embed[input_type] = {}
            for input_name, (input_loc, input_size) in inputs.items():
                if input_size == 0:
                    self._embed[input_type][input_loc] = LinearUnit(self.d_model, None, use_time_distributed)
                else:
                    self._embed[input_type][input_loc] = EmbeddingUnit(self.d_model, input_size, use_time_distributed)
        
        self._static_grn_cvs = GatedResidualNetwork(self.d_model, False, False, None, self.dropout_rate)
        self._static_grn_ce = GatedResidualNetwork(self.d_model, False, False, None, self.dropout_rate)
        self._static_grn_ch = GatedResidualNetwork(self.d_model, False, False, None, self.dropout_rate)
        self._static_grn_cc = GatedResidualNetwork(self.d_model, False, False, None, self.dropout_rate)

        self._lstm_observed = krs.layers.LSTM(
            self.d_model,
            return_sequences=True,
            return_state=True,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        )
        self._lstm_forecast = krs.layers.LSTM(
            self.d_model,
            return_sequences=True,
            return_state=False,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        )

        self._glu_temporal = GatedLinearUnit(self.d_model, None, True, self.dropout_rate)
        self._ann_temporal = AddAndNormalize()

        self._static_ce_grn = GatedResidualNetwork(self.d_model, True, True, None, self.dropout_rate)
        # self._self_attention = krs.layers.MultiHeadAttention(num_attn_heads, self.d_model, self.d_model, self.dropout_rate, False)
        self._self_attention = InterpretableMultiHeadAttention(num_attn_heads, self.d_model, True, self.dropout_rate)

        self._glu_attention = GatedLinearUnit(self.d_model, None, True, self.dropout_rate)
        self._ann_attention = AddAndNormalize()

        self._pwff_grn = GatedResidualNetwork(self.d_model, True, False, None, self.dropout_rate)

        self._glu_final = GatedLinearUnit(self.d_model, None, True, None)
        self._ann_final = AddAndNormalize()

        self._linear = LinearUnit(len(self._data_specs['targets'])*len(self._quantile_spec), None, True, True)

    # def call(self, inputs: Dict[str, Dict[str, tf.Tensor]], training=None, mask=None):
        # static_inputs = []
        # observed_inputs = []
        # forecast_inputs = []

    def call(self, static_inputs, observed_inputs, forecast_inputs, training=None, mask=None):
        # static_inputs, observed_inputs, forecast_inputs = inputs
        static_embedding_layer = []
        observed_embedding_layer = []
        forecast_embedding_layer = []

        for input_loc, embeding in self._embed['static'].items():
            tensor_input = tf.gather(static_inputs, [input_loc], axis=-1)
            static_embedding_layer.append(embeding(tensor_input))
        for input_loc, embeding in self._embed['observed'].items():
            tensor_input = tf.gather(static_inputs, [input_loc], axis=-1)
            observed_embedding_layer.append(embeding(tensor_input))
        for input_loc, embeding in self._embed['forecast'].items():
            tensor_input = tf.gather(static_inputs, [input_loc], axis=-1)
            forecast_embedding_layer.append(embeding(tensor_input))

        static_embedding_layer = krs.backend.stack(static_embedding_layer, axis=-1)
        observed_embedding_layer = krs.backend.stack(observed_embedding_layer, axis=-1)
        forecast_embedding_layer = krs.backend.stack(forecast_embedding_layer, axis=-1)

        static_encoder, static_flags = self._vsn['static'](static_inputs)
        static_cvs, _ = self._static_grn_cvs(static_encoder)
        static_ce, _ = krs.backend.expand_dims(self._static_grn_ce(static_encoder), 1)
        static_ch, _ = self._static_grn_ch(static_encoder)
        static_cc, _ = self._static_grn_cc(static_encoder)

        observed_features, observed_flags = self._vsn['observed'](observed_inputs, static_cvs)
        forecast_features, forecast_flags = self._vsn['forecast'](forecast_inputs, static_cvs)

        feature_layer = krs.backend.concatenate([observed_features, forecast_features], axis=1)

        observed_lstm, state_h, state_c = \
            self._lstm_observed(observed_features, initial_state=[static_ch, static_cc])
        forecast_lstm = \
            self._lstm_forecast(forecast_features, initial_state=[state_h, state_c])

        lstm_layer = krs.backend.concatenate([observed_lstm, forecast_lstm], axis=1)
        
        gated_lstm_layer, _ = self._glu_temporal(lstm_layer)
        temporal_feature_layer = self._ann_temporal([gated_lstm_layer, feature_layer])
        enriched_temporal_layer = self._static_ce_grn(temporal_feature_layer, static_ce)

        mask_layer = krs.backend.cumsum(tf.eye(
            krs.backend.shape(enriched_temporal_layer)[1],
            batch_shape=krs.backend.shape(enriched_temporal_layer)[:1]
        ), 1)

        weight_layer, self_attention_flags = self._self_attention(
            enriched_temporal_layer,
            enriched_temporal_layer,
            enriched_temporal_layer,
            mask_layer
        )

        gated_weight_layer, _ = self._glu_attention(weight_layer)
        ann_weight_layer = self._ann_attention([gated_weight_layer, enriched_temporal_layer])

        temporal_decoder_layer, _ = self._pwff_grn(ann_weight_layer)

        gated_temporal_layer = self._glu_final(temporal_decoder_layer)
        ann_temporal_layer = self._ann_final([gated_temporal_layer, temporal_feature_layer])

        forecast_size = krs.backend.shape(forecast_inputs)[-2]
        target_quantiles = self._linear(ann_temporal_layer[..., -forecast_size:, :])

        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_attention_flags,
            # Static variable selection weights
            'static_flags': static_flags,
            # Variable selection weights of past inputs
            'historical_flags': observed_flags,
            # Variable selection weights of future inputs
            'future_flags': forecast_flags
        }

        return target_quantiles, attention_components


    def loss(self, targets, predictions):

        self._quantile_spec
        output_size = len(self._data_specs['targets'])

        def quantile_loss(y, yp, q):
            q_loss = q * (y - yp) + (1 - q) * (yp - y)
            return tf.reduce_sum(q_loss, axis=-1)

        loss = 0
        for i, q in enumerate(self._quantile_spec):
            loss += quantile_loss(
                targets[..., output_size*i:output_size*(i+1)],
                predictions[..., output_size*i:output_size*(i+1)],
                q
            )

        return loss

    def compile(self, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        krs.Model().compile()
        
        adam = tf.compat.v1.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=self.max_gradient_norm)
        return super().compile(adam, self.loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
