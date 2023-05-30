
import os
import arrow
import tensorflow as tf
import tensorflow_io as tfio
from ai.tft2.attention import InterpretableMultiHeadSelfAttention
from ai.tft2.embedding import GenericEmbedding
from ai.tft2.gating import GatedResidualNetwork, VariableSelectionNetwork, drop_gate_skip_norm
from resources.enums import BarColumn
from resources.time_tools import ClockController

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
        @tf.function
        def get_weekday(ts):
            return arrow.get(ts, tzinfo=ClockController.time_zone).weekday()

        input_spec = {
            'static': {
                'day_of_week': {
                    'num_categories': 5,
                    'input_transform': lambda fname, data: get_weekday(fname)
                }
            },
            'observed': {
                'time_of_day_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.TimeStamp.value]
                },
                'volume_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.Volume.value]
                },
                'open_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.Open.value]
                },
                'high_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.High.value]
                },
                'low_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.Low.value]
                },
                'close_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.Close.value]
                },
                'vwap_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.VWAP.value]
                }
            },
            'forecast': {
                'time_of_day_ts': {
                    'num_categories': 0,
                    'input_transform': lambda fname, data: data[BarColumn.TimeStamp.value]
                }
            }
        }
        target_spec = {
            'high_ts': {
                'num_categories': 0,
                'quantiles': [0.1, 0.25, 0.5],
                'input_transform': lambda fname, data: data[BarColumn.High.value]
            },
            'low_ts': {
                'num_categories': 0,
                'quantiles': [0.5, 0.75, 0.9],
                'input_transform': lambda fname, data: data[BarColumn.Low.value]
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
    
    def dataset_generator(self, data_path, batch_size, pct_tvt):
        num_days = len(os.listdir(f"{data_path}/../metadata"))
        
        full_dataset = tf.data.Dataset.list_files(f"{data_path}/*.csv", shuffle=True)

        def make_dataset(dataset: tf.data.Dataset, model_name: str):
            WINDOW_LEN = self.lookback + self.lookforward

            def csv_to_dataset(fname):
                NUMERIC_TYPE = tf.float64
                REPEAT = self.lookforward - 1

                day_ts = tf.strings.to_number(
                    tf.strings.split(
                        tf.strings.split(
                            fname,
                            os.path.sep
                        )[-1],
                        '.'
                    )[0],
                    out_type=NUMERIC_TYPE
                )
                
                meta = tfio.experimental.serialization.decode_json(
                    tf.io.read_file(tf.strings.format('data/metadata/{}.json', day_ts)), 
                    specs={
                        "day_of_week": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE),
                        "bod_ts": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE),
                        "eod_ts": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE)
                    }
                )

                fcsv = tf.io.read_file(fname)
                to_lines = tf.strings.split(fcsv, '\r\n')[:-1]
                to_array = tf.strings.split(to_lines, ',')
                to_tensor_series = tf.strings.to_number(to_array, out_type=NUMERIC_TYPE).to_tensor()

                time_delta = to_tensor_series[-1, 0] - to_tensor_series[-2, 0]
                ts = to_tensor_series[-1, 0] + tf.range(time_delta, time_delta*(REPEAT+1), time_delta)
                price = to_tensor_series[-1, 4] * tf.ones([REPEAT, 2], dtype=NUMERIC_TYPE)
                
                repeated_tensor = tf.concat([
                    tf.expand_dims(ts, -1),
                    tf.zeros([REPEAT, 1], dtype=NUMERIC_TYPE),
                    price,
                    tf.zeros([REPEAT, 4], dtype=NUMERIC_TYPE)
                ], -1)

                return meta, tf.concat([to_tensor_series, repeated_tensor], 0)

            def batch_window_len(data: tf.data.Dataset):
                return data.batch(WINDOW_LEN, drop_remainder=True)
            
            dataset = dataset.map(csv_to_dataset)
            dataset = dataset.map(
                lambda m, t: 
                (m, tf.data.Dataset.from_tensor_slices(t).window(WINDOW_LEN, 1).flat_map(batch_window_len))
            )

            def ds_to_spec(dataset):
                for meta, window_data in dataset:
                    for data in window_data:
                        inputs=(
                            tf.expand_dims(meta["day_of_week"], -1),
                            tf.expand_dims(data[:self.lookback, 0]-meta["bod_ts"], -1),
                            tf.expand_dims(data[:self.lookback, 5], -1),
                            tf.expand_dims(data[:self.lookback, 1], -1),
                            tf.expand_dims(data[:self.lookback, 2], -1),
                            tf.expand_dims(data[:self.lookback, 3], -1),
                            tf.expand_dims(data[:self.lookback, 4], -1),
                            tf.expand_dims(data[:self.lookback, 7], -1),
                            tf.expand_dims(data[self.lookback:, 0]-meta["bod_ts"], -1)
                        )
                        outputs=(
                            tf.expand_dims(data[self.lookback:, 2], -1),
                            tf.expand_dims(data[self.lookback:, 3], -1)
                        )
                        yield inputs, outputs
            
            return tf.data.Dataset.from_generator(
                lambda: ds_to_spec(dataset),
                output_signature=(
                    (
                        tf.TensorSpec(shape=[1], dtype=tf.int32),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookback, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookforward, 1], dtype=tf.float64)
                    ),
                    (
                        tf.TensorSpec(shape=[self.lookforward, 1], dtype=tf.float64),
                        tf.TensorSpec(shape=[self.lookforward, 1], dtype=tf.float64)
                    )
                )
            ).cache(f"{data_path}/{model_name}").shuffle(10*batch_size).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        train_dataset = full_dataset.take(round(num_days*pct_tvt[0]))
        self.train_dataset = make_dataset(train_dataset, 'train')
        if pct_tvt[1] > 0:
            validation_dataset = full_dataset.skip(round(num_days*pct_tvt[0])).take(round(num_days*(pct_tvt[1]-pct_tvt[0])))
            self.validation_dataset = make_dataset(validation_dataset, 'validate')
        if pct_tvt[1] < 1:
            test_dataset = full_dataset.skip(round(num_days*pct_tvt[1]))
            self.test_dataset = make_dataset(test_dataset, 'test')

    # @tf.function
    # def fit(self, dataset):
    #     def ds_to_spec(fname, data):
    #         inputs=[
    #             input_data['input_transform'](fname, data)
    #             for _, input_type in self.input_spec.items()
    #             for _, input_data in input_type.items()
    #         ]
    #         outputs=[
    #             target_data['input_transform'](fname, data)
    #             for _, target_data in self.target_spec.items()
    #         ]
    #         return inputs, outputs
        
    #     for day_ts, day_dataset in dataset:
    #         for window_data in day_dataset:
    #             inputs, outputs = ds_to_spec(day_ts, window_data)
    #             predictions = self.model.predict(inputs)


if __name__ == "__main__":

    d_model = 64
    d_att = 5
    lb = 100
    lf = 10

    chkpnt_path = f"data/1_min/model-{d_model}-{d_att}-{lb}-{lf}-" + "{epoch:04d}.ckpt"
    chkpnt_dir = os.path.dirname(chkpnt_path)

    latest_chkpnt = tf.train.latest_checkpoint(chkpnt_dir)

    tft = TemporalFusionTransformer(None, None, d_model, d_att, lb, lf, 0.2)
    tft.dataset_generator('data/1_min', 64, [0.8, 1.0])
    tft.model.compile(optimizer=tf.optimizers.Adam(), loss=tft.quantile_loss)
    tft.model.summary()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpnt_path,
        save_weights_only=True,
        verbose=1
    )

    tft.model.fit(
        tft.train_dataset,
        validation_data=tft.validation_dataset,
        epochs=25,
        callbacks=[cp_callback]
    )

    exit()
