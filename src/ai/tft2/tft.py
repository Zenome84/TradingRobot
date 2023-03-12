
import tensorflow as tf
from ai.tft2.attention import InterpretableMultiHeadSelfAttention
from ai.tft2.embedding import GenericEmbedding
from ai.tft2.gating import GatedResidualNetwork, VariableSelectionNetwork, drop_gate_skip_norm

layerLSTM = tf.keras.layers.LSTM
layerInput = tf.keras.layers.Input
layerDense = tf.keras.layers.Dense
layerTimeDistributed = tf.keras.layers.TimeDistributed

class TemporalFusionTransformer:
    def __init__(self, input_spec, target_spec, d_model, att_heads, lookback, lookforward, dropout_rate):
        '''
        input_spec: of the form
            {
                'static': {
                    'var_name': { 'num_categories': Positive Integer or 0 for Reals },
                    ...
                },
                'observed': { Same Structure },
                'forecast': { Same Structure }
            }

        target_spec: of the form (at the moment only supports Reals)
            {
                'var_name': { 'quantiles': [List of quantiles] },
                ...
            }

        d_model: Embedding feature size to use throughout the TFT model

        att_heads: Number of attention heads to use

        lookback: timeseries length of 'observed' values

        lookforward: timeseries length of 'forecast' values

        dropout_rate: dropout rate to use during training
        '''
        input_spec = {
            'static': {
                'day_of_week': {
                    'num_categories': 5
                }
            },
            'observed': {
                'time_of_day_ts': {
                    'num_categories': 0
                },
                'volume_ts': {
                    'num_categories': 0
                },
                'open_ts': {
                    'num_categories': 0
                },
                'high_ts': {
                    'num_categories': 0
                },
                'low_ts': {
                    'num_categories': 0
                },
                'close_ts': {
                    'num_categories': 0
                },
                'vwap_ts': {
                    'num_categories': 0
                }
            },
            'forecast': {
                'time_of_day_ts': {
                    'num_categories': 0
                }
            }
        }
        target_spec = {
            'high_ts': {
                'num_categories': 0,
                'quantiles': [0.1, 0.5, 0.9]
            },
            'low_ts': {
                'num_categories': 0,
                'quantiles': [0.1, 0.5, 0.9]
            }
        }

        self.input_spec = input_spec
        self.target_spec = target_spec
        self.d_model = d_model
        self.att_heads = att_heads
        self.lookback = lookback
        self.lookforward = lookforward
        self.dropout_rate = dropout_rate

        for input_name, input_data in self.input_spec['static'].items():
            self.input_spec['static'][input_name]['input_tensor'] = layerInput(shape=[1])
            self.input_spec['static'][input_name]['embedding_tensor'] = \
                GenericEmbedding(
                    num_categories=input_data['num_categories'],
                    embedding_size=d_model
                )(input_data['input_tensor'])
        for input_name, input_data in self.input_spec['observed'].items():
            self.input_spec['observed'][input_name]['input_tensor'] = layerInput(shape=[self.lookback, 1])
            self.input_spec['observed'][input_name]['embedding_tensor'] = \
                layerTimeDistributed(GenericEmbedding(
                    num_categories=input_data['num_categories'],
                    embedding_size=d_model
                ))(input_data['input_tensor'])
        for input_name, input_data in self.input_spec['forecast'].items():
            self.input_spec['forecast'][input_name]['input_tensor'] = layerInput(shape=[self.lookforward, 1])
            self.input_spec['forecast'][input_name]['embedding_tensor'] = \
                layerTimeDistributed(GenericEmbedding(
                    num_categories=input_data['num_categories'],
                    embedding_size=d_model
                ))(input_data['input_tensor'])
        
        static_encoder_tensor, self.static_flags = VariableSelectionNetwork(
            num_features=len(self.input_spec['static']),
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )([
            input_data['embedding_tensor']
            for _, input_data in self.input_spec['static'].items()
        ])

        cvs_tensor = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )(static_encoder_tensor)
        ce_tensor = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )(static_encoder_tensor)
        ch_tensor = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )(static_encoder_tensor)
        cc_tensor = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )(static_encoder_tensor)

        observed_features, self.observed_flags = layerTimeDistributed(
            VariableSelectionNetwork(
                num_features=len(self.input_spec['observed']),
                units=self.d_model,
                dropout_rate=self.dropout_rate,
                with_context=True
        ))([
            input_data['embedding_tensor']
            for _, input_data in self.input_spec['observed'].items()
        ] + [tf.repeat(tf.expand_dims(cvs_tensor, 1), self.lookback, 1)])
        
        forecast_features, self.forecast_flags = layerTimeDistributed(
            VariableSelectionNetwork(
                num_features=len(self.input_spec['forecast']),
                units=self.d_model,
                dropout_rate=self.dropout_rate,
                with_context=True
        ))([
            input_data['embedding_tensor']
            for _, input_data in self.input_spec['forecast'].items()
        ] + [tf.repeat(tf.expand_dims(cvs_tensor, 1), self.lookforward, 1)])

        observed_lstm, state_h, state_c = layerLSTM(
            self.d_model,
            return_sequences=True,
            return_state=True,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        )(observed_features, initial_state=[ch_tensor, cc_tensor])
        forecast_lstm = layerLSTM(
            self.d_model,
            return_sequences=True,
            return_state=False,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        )(forecast_features, initial_state=[state_h, state_c])

        feature_layer = tf.keras.layers.concatenate([observed_features, forecast_features], axis=1)
        lstm_layer = tf.keras.layers.concatenate([observed_lstm, forecast_lstm], axis=1)

        temporal_feature_layer = drop_gate_skip_norm(lstm_layer, feature_layer, self.dropout_rate)
        enriched_temporal_layer = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )([temporal_feature_layer] + [tf.repeat(tf.expand_dims(ce_tensor, 1), self.lookback + self.lookforward, 1)])

        attention_layer, self.attention_scores = InterpretableMultiHeadSelfAttention(
            self.att_heads, self.d_model, self.dropout_rate)(enriched_temporal_layer)

        temporal_attention = drop_gate_skip_norm(
            attention_layer[..., -self.lookforward:, :],
            enriched_temporal_layer[..., -self.lookforward:, :],
            self.dropout_rate
        )

        temporal_decoder = GatedResidualNetwork(
            units=self.d_model,
            dropout_rate=self.dropout_rate
        )(temporal_attention)

        final_layer = drop_gate_skip_norm(
            temporal_decoder,
            temporal_feature_layer[..., -self.lookforward:, :],
            self.dropout_rate
        )

        for target_name, target_data in self.target_spec.items():
            self.target_spec[target_name]['target_tensor'] = layerDense(len(target_data['quantiles']))(final_layer)

        self.model = tf.keras.Model(
            inputs=[
                input_data['input_tensor']
                for _, input_type in self.input_spec.items()
                for _, input_data in input_type.items()
            ],
            outputs=[
                target_data['target_tensor']
                for _, target_data in self.target_spec.items()
            ]
        )

    @tf.function
    def quantile_loss(self, targets, predictions):
        cumulative_loss = tf.constant(0.)
        for target_idx, target_data in enumerate(self.target_spec.values()):
            prediction_errors = targets[target_idx] - predictions[target_idx]
            quantiles = tf.expand_dims(tf.constant(target_data['quantiles']), -1)
            cumulative_loss += tf. reduce_mean(tf.matmul(tf.nn.relu(prediction_errors), quantiles) \
                + tf.matmul(tf.nn.relu(-prediction_errors), (1 - quantiles)))
        return cumulative_loss
    
    def fit(self, inputs, outputs):
        ...

        # self.model.fit([
        #     np.expand_dims(input_data, 0)
        #     for _, input_type in inputs.items()
        #     for _, input_data in input_type.items()
        # ], lo)