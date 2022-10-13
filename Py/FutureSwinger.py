
from __future__ import annotations

import concurrent.futures
import datetime
import gc
import glob
import json
import math
import os
import time
import psutil
from random import shuffle
from typing import Any, Dict

import arrow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
import tensorflow.keras.layers as lyr
import tensorflow_io as tfio
import tensorflow_transform as tft
from core.robot_client import RobotClient
from resources.enums import BarSize
from resources.time_tools import ClockController, wait_until

# path = "U:/MarketData/ES_GLOBEX/HLS/"
# fnames = glob.glob(f"{path}*")
# for fname in fnames:
#     ffnames = glob.glob(f"{fname}/*")
#     new_fname = f"{os.path.sep.join(str.split(fname,os.path.sep)[:-1])}/{min(ffnames).split(os.path.sep)[-1].split('.')[0]}.{max(ffnames).split(os.path.sep)[-1].split('.')[0]}"
#     os.rename(fname, new_fname)

# exit(0)

def tf_accumulate(input_tensor, func):
    def tf_while_condition(input_tensor, loop_counter):
        return tf.not_equal(loop_counter, 0)

    def tf_while_body(input_tensor, loop_counter):
        loop_counter -= 1
        if func == 'max':
            new_input_tensor = tf.maximum(input_tensor, tf.concat(([input_tensor[0]], input_tensor[:-1]), axis=0))
        elif func == 'min':
            new_input_tensor = tf.minimum(input_tensor, tf.concat(([input_tensor[0]], input_tensor[:-1]), axis=0))
        else:
            raise NotImplementedError("Please implement an accumulator")
        return new_input_tensor, loop_counter

    return tf.while_loop(
        cond=tf_while_condition, 
        body=tf_while_body, 
        loop_vars=(input_tensor, tf.shape(input_tensor)[0])
    )[0]


MAX_DAYS = 30
INPUT_DATA_SPECS = {
    BarSize.SEC_05: min(MAX_DAYS*24*60*2, 300),
    BarSize.SEC_15: min(MAX_DAYS*24*60*2, 300),
    BarSize.SEC_30: min(MAX_DAYS*24*60*2, 300),
    BarSize.MIN_01: min(MAX_DAYS*24*60, 250),
    BarSize.MIN_05: min(MAX_DAYS*24*12, 200),
    BarSize.MIN_15: min(MAX_DAYS*24*4, 150),
    BarSize.MIN_30: min(MAX_DAYS*24*2, 120),
    BarSize.HRS_01: min(MAX_DAYS*24, 90),
    BarSize.HRS_04: min(MAX_DAYS*6, 60),
    BarSize.DAY_01: min(MAX_DAYS, 30)
}
OUTPUT_DATA_SPECS = {
    BarSize.SEC_05: {"Bar": BarSize.SEC_05, "Length": 5//5},
    BarSize.SEC_15: {"Bar": BarSize.SEC_05, "Length": 15//5},
    BarSize.SEC_30: {"Bar": BarSize.SEC_05, "Length": 30//5},
    BarSize.MIN_01: {"Bar": BarSize.SEC_05, "Length": 60//5},
    BarSize.MIN_05: {"Bar": BarSize.SEC_05, "Length": 60//5*5},
    BarSize.MIN_10: {"Bar": BarSize.SEC_05, "Length": 60//5*10},
    BarSize.MIN_15: {"Bar": BarSize.SEC_05, "Length": 60//5*15},
    BarSize.MIN_30: {"Bar": BarSize.SEC_15, "Length": 60//5*30//3},
    BarSize.HRS_01: {"Bar": BarSize.SEC_15, "Length": 60//5*60//3},
    BarSize.HRS_02: {"Bar": BarSize.SEC_30, "Length": 60//5*60*2//6}
}

NUMERIC_TYPE = tf.float32
FILE_SPECS = {
    "inputData": {
        bar.value: tf.TensorSpec(tf.TensorShape([None, 8]), NUMERIC_TYPE, name=f"inputData_{bar.name}")
        for bar in INPUT_DATA_SPECS.keys()
    },
    "matchData": {
        bar.value: {
            "high": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE, name=f"matchData_{bar.name}_high"),
            "low": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE, name=f"matchData_{bar.name}_low")
        }
        for bar in OUTPUT_DATA_SPECS.keys()
    }
}
# VERIFY_SPECS = {
#     "inputData": {
#         bar.value: tf.TensorSpec(tf.TensorShape([None, 8]), NUMERIC_TYPE, name=f"inputData_{bar.name}")
#         for bar, bar_length in INPUT_DATA_SPECS.items()
#     },
#     "matchData": {
#         bar.value: {
#             "high": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE, name=f"matchData_{bar.name}_high"),
#             "low": tf.TensorSpec(tf.TensorShape([]), NUMERIC_TYPE, name=f"matchData_{bar.name}_low")
#         }
#         for bar in OUTPUT_DATA_SPECS.keys()
#     }
# }
class FutureSwinger(krs.Model):
    def __init__(self):
        super(FutureSwinger, self).__init__()
        self.lstm ={
            bar.value: lyr.LSTM(4, return_sequences=True, return_state=True, name=f"LSTM_{bar.name}")
            for bar, bar_length in INPUT_DATA_SPECS.items()
        }


        # self.conv1 = {
        #     bar.value: lyr.Conv1D(5, bar_length//10, activation='relu', use_bias=True, name=f"CONV1_{bar.name}")
        #     for bar, bar_length in INPUT_DATA_SPECS.items()
        # }
        # self.conv2 = {
        #     bar.value: lyr.Conv1D(10, bar_length - bar_length//10 + 1, activation='relu', use_bias=True, name=f"CONV2_{bar.name}")
        #     for bar, bar_length in INPUT_DATA_SPECS.items()
        # }
        # self.flatten = {
        #     bar.value: lyr.Flatten(name=f"FLAT_{bar.name}")
        #     for bar in INPUT_DATA_SPECS.keys()
        # }

        # self.dense = {
        #     bar.value: lyr.Dense(10, activation='relu', use_bias=True, name=f"DENSE_{bar.name}")
        #     for bar in INPUT_DATA_SPECS.keys()
        # }
        self.dense_tod = lyr.Dense(20, activation='relu', use_bias=True, name=f"DENSE_TOD")

        # self.dense_high1 = lyr.Dense(64, activation='relu', use_bias=True, name=f"DENSE_HIGH1")
        # self.dense_low1 = lyr.Dense(64, activation='relu', use_bias=True, name=f"DENSE_LOW1")
        # self.dense_high2 = lyr.Dense(32, activation='relu', use_bias=True, name=f"DENSE_HIGH2")
        # self.dense_low2 = lyr.Dense(32, activation='relu', use_bias=True, name=f"DENSE_LOW2")
        # self.full_high = lyr.Dense(10, activation='sigmoid', use_bias=True, name=f"FULL_HIGH")
        # self.full_low = lyr.Dense(10, activation='sigmoid', use_bias=True, name=f"FULL_LOW")

        self.dense_high = {
            bar.value: lyr.Dense(32, activation='relu', use_bias=True, name=f"DENSE_HIGH_{bar.name}")
            for bar in OUTPUT_DATA_SPECS.keys()
        }
        self.dense_low = {
            bar.value: lyr.Dense(32, activation='relu', use_bias=True, name=f"DENSE_LOW_{bar.name}")
            for bar in OUTPUT_DATA_SPECS.keys()
        }
        self.full_high = {
            bar.value: lyr.Dense(64, activation='softmax', use_bias=True, name=f"FULL_HIGH_{bar.name}")
            for bar in OUTPUT_DATA_SPECS.keys()
        }
        self.full_low = {
            bar.value: lyr.Dense(64, activation='softmax', use_bias=True, name=f"FULL_LOW_{bar.name}")
            for bar in OUTPUT_DATA_SPECS.keys()
        }
    
    def call(self, inputs):
        bar_data = inputs[:, :-1]
        lstm_out = {}
        lstm_error = {}
        cell_out = {}
        start_idx = 0
        # for bar, bar_length in INPUT_DATA_SPECS.items():
        #     conv_out[bar.value] = (
        #         # self.dense[bar.value](
        #             self.flatten[bar.value](
        #                 self.conv2[bar.value](
        #                     self.conv1[bar.value](
        #                         bar_data[:, start_idx:start_idx+bar_length]
        #                     )
        #                 )
        #             )
        #         # )
        #     )
        #     start_idx += bar_length
        for bar, bar_length in INPUT_DATA_SPECS.items():
            lstm_out[bar.value], _, cell_out[bar.value] = (
                self.lstm[bar.value](
                    bar_data[:, start_idx:start_idx+bar_length]
                )
            )
            lstm_error[bar.value] = tf.math.reduce_sum(tf.math.square(lstm_out[bar.value][:, :-1] - bar_data[:, start_idx+1:start_idx+bar_length, :-1]))
            start_idx += bar_length

        bar_out = tf.concat([c_out for c_out in cell_out.values()], axis=-1)
        self.lstm_mse = tf.math.sqrt(tf.math.reduce_sum([err for err in lstm_error.values()])/bar_data.shape[1])

        tod = inputs[:, -1:0, 0]
        tod_out = self.dense_tod(tod)
        combined_out = tf.concat([bar_out, tod_out], axis=-1)

        high_out = tf.concat([
            tf.expand_dims(tf.expand_dims(
                self.full_high[bar.value](
                    self.dense_high[bar.value](
                        combined_out
                    )
                ), axis=-2
            ), axis=-2)
            for bar in OUTPUT_DATA_SPECS.keys()
        ], axis=-3)
        low_out = tf.concat([
            tf.expand_dims(tf.expand_dims(
                self.full_low[bar.value](
                    self.dense_low[bar.value](
                        combined_out
                    )
                ), axis=-2
            ), axis=-2)
            for bar in OUTPUT_DATA_SPECS.keys()
        ], axis=-3)
        outputs = tf.concat([
            high_out,
            low_out
        ], axis=-2)
        
        # high_out = tf.expand_dims((
        #     self.full_high(
        #         self.dense_high2(
        #             self.dense_high1(
        #                 combined_out
        #             )
        #         )
        #     )
        # ), axis=-1)
        # low_out = tf.expand_dims((
        #     self.full_low(
        #         self.dense_low2(
        #             self.dense_low1(
        #                 combined_out
        #             )
        #         )
        #     )
        # ), axis=-1)

        # outputs = tf.concat([
        #     1 + tf.math.cumsum(high_out/50, axis=-2),
        #     1 - tf.math.cumsum(low_out/50, axis=-2)
        # ], axis=-1)
        return outputs
        
    @staticmethod
    def read_file(fpath):
        fpath_split = tf.strings.split(fpath, os.path.sep)
        
        ts = tf.strings.to_number(tf.strings.split(fpath_split[-1], '.')[0], out_type=NUMERIC_TYPE)
        am_pm = tf.strings.split(fpath_split[-2], '.')
        am_ts = tf.strings.to_number(am_pm[0], out_type=NUMERIC_TYPE)
        pm_ts = tf.strings.to_number(am_pm[1], out_type=NUMERIC_TYPE)

        return (tfio.experimental.serialization.decode_json(tf.io.read_file(fpath), specs=FILE_SPECS), (ts - am_ts) / (pm_ts - am_ts))

    @staticmethod
    def verify_file(json_obj, ts_obj):
        verify = tf.concat([
            [tf.cast(tf.not_equal(tf.shape(json_obj["inputData"][bar.value]), tf.TensorShape([bar_length, 8])), tf.int32)]
            for bar, bar_length in INPUT_DATA_SPECS.items()
        ], axis = 0)
        return tf.equal(0, tf.reduce_sum(verify))

    @staticmethod
    def process_file(json_obj, ts_obj):
        dict_obj = json_obj
        dict_obj["inputData"]["timeOfDay"] = ts_obj
        dict_obj["inputData"]["currPrice"] = json_obj["inputData"][BarSize.SEC_05.value][-1][4]

        return dict_obj

    @staticmethod
    def preprocess_dict(dict_obj):
        currPrice = dict_obj["inputData"]["currPrice"]
        tod = dict_obj["inputData"]["timeOfDay"]
        
        inputData = tf.concat([
            tf.concat([
                dict_obj["inputData"][bar.value][:, 1:5]/currPrice,
                (dict_obj["inputData"][bar.value][:, 5:6] - tf.reduce_mean(dict_obj["inputData"][bar.value][:, 5:6])) \
                    / tf.math.reduce_std(dict_obj["inputData"][bar.value][:, 5:6])
            ], axis=1)
            for bar in INPUT_DATA_SPECS.keys()
        ], axis=0)
        inputData = tf.concat([inputData, [[tod, currPrice, 0, 0, 0]]], axis=0)

        buckets = np.array([range(63)])/63*0.012
        matchData = tf.transpose(tf.concat([
            [[val["high"], val["low"]]]
            for val in dict_obj["matchData"].values()
        ], axis=0))
        # matchData = tf.transpose(tf.concat([
        #     [tf_accumulate(matchData[0], 'max')/currPrice], 
        #     [tf_accumulate(matchData[1], 'min')/currPrice]
        # ], axis=0))
        matchData = tf.transpose(tf.concat([
            [tft.apply_buckets(
                tf_accumulate(matchData[0], 'max')/currPrice,
                1 + buckets
            )], 
            [63 - tft.apply_buckets(
                tf_accumulate(matchData[1], 'min')/currPrice,
                np.flip(1 - buckets)
            )]
        ], axis=0))
        matchData = tf.one_hot(matchData, 64)
        return (inputData, matchData)


if __name__ == "__main__":
    # data_path = "U:/MarketData/ES_GLOBEX/HLS/"
    data_path = "S:/HLS/"
    train_label = "train"
    valid_label = "valid"

    train_dataset = tf.data.Dataset.list_files(f"{data_path}{train_label}/*/*.json", shuffle=True)
    test_dataset = tf.data.Dataset.list_files(f"{data_path}{valid_label}/*/*.json", shuffle=True)

    def preprocess(dataset, name):
        return dataset.map(FutureSwinger.read_file).filter(FutureSwinger.verify_file).map(FutureSwinger.process_file).map(FutureSwinger.preprocess_dict).cache(f"{data_path}{name}.cache").repeat(1).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    batch_size = 2**12
    total_size = 481*4679
    train_size = 441*4679
    valid_size = 40*4679
    print(f"total Size: {total_size} | Training Size: {train_size} | Validation Size: {valid_size}")
    train_ds = preprocess(train_dataset, train_label)
    valid_ds = preprocess(test_dataset, valid_label)
    # test_ds = preprocess(dataset_for_model.skip(train_size).skip(val_size))

    model = FutureSwinger()
    
    train_loss_results = []
    train_accuracy_results = []
    valid_loss_results = []
    valid_accuracy_results = []

    num_epochs = 200
    optimizer = tf.keras.optimizers.Adam()
    def loss(model, x, y, training):
        y_ = model(x, training=training)
        return krs.losses.CategoricalCrossentropy()(y_true=y, y_pred=y_) + model.lstm_mse, y_
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value, y_ = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables), y_

    first_time = True
    for epoch in range(num_epochs):

        i = 0
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        # Training loop
        for x, y in train_ds:
            # Optimize the model
            loss_value, grads, y_pred = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if first_time:
                model.summary()
                first_time = False

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, y_pred)

            j = (i + 1) / train_size * batch_size
            i += 1
            print("Training Data: [%-20s] %d%%" %
                  ('='*int(20*j), 100*j), end='\r')
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results.append(epoch_accuracy.result())
        str_train = f"Epoch {epoch:03d}: Training (Loss, Accuracy): {epoch_loss_avg.result():.3f}, {epoch_accuracy.result():.3%}"

        i = 0
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        # Validation loop
        for x, y in valid_ds:
            # Get loss
            loss_value, y_pred = loss(model, x, y, training=False)

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, y_pred)

            j = (i + 1) / valid_size * batch_size
            i += 1
            print("Validating Data: [%-20s] %d%%" %
                  ('='*int(20*j), 100*j), end='\r')
        # End epoch
        valid_loss_results.append(epoch_loss_avg.result())
        # valid_accuracy_results.append(epoch_accuracy.result())


        print(f"{str_train} | Validation (Loss, Accuracy): {epoch_loss_avg.result():.3f}, {epoch_accuracy.result():.3%}")

    # model.compile(
    #     optimizer='adam',
    #     loss=krs.losses.CategoricalCrossentropy(),
    #     metrics=['accuracy']
    # )
    # history = model.fit(train_ds, validation_data=valid_ds, epochs=200)

    buckets = np.array([[0] + list(range(63))])/63*0.012
    buckets = np.concatenate([1 + buckets, (1 - buckets)], axis=0)
    for input_batch, output_batch in valid_ds.take(5):
        pred = model.predict(input_batch)
        # plt.plot([5, 15, 30, 60, 60*5, 60*10, 60*15, 60*30, 60*60, 60*60*2], np.sum(output_batch[0].numpy() * buckets, axis=-1))
        # plt.plot([5, 15, 30, 60, 60*5, 60*10, 60*15, 60*30, 60*60, 60*60*2], np.sum(pred[0] * buckets, axis=-1))
        plt.plot([5, 15, 30, 60, 60*5, 60*10, 60*15, 60*30, 60*60, 60*60*2], output_batch[0].numpy())
        plt.plot([5, 15, 30, 60, 60*5, 60*10, 60*15, 60*30, 60*60, 60*60*2], pred[0])
        plt.show()
        print('next')
    
    print('Done')


exit(0)

if __name__ == "__main__":
    path = "U:/MarketData/ES_GLOBEX/HLS/"
    fnames = glob.glob(f"{path}*")

    def extract_dictionary(fname: str):
        print(f"Loading: {fname}")
        with open(fname, 'r') as file:
            day_dict = json.load(file)

        day_path = f"{path}{fname[-13:-5]}"
        os.mkdir(day_path)

        n = len(day_dict)
        i = 0
        for ts, data in day_dict.items():
            with open(f"{day_path}/{ts}.json", 'w') as file:
                json.dump(data, file)

            j = (i + 1) / n
            i += 1
            print("Extracting dictionary: [%-20s] %d%%" %
                  ('='*int(20*j), 100*j), end='\r')

        print(f"Deleting: {fname}")
        os.remove(fname)

    for fname in fnames:
        extract_dictionary(fname)
        gc.collect()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = []
    #     for fname in fnames:
    #         futures.append(executor.submit(extract_dictionary, fname=fname))
    #     for future in concurrent.futures.as_completed(futures):
    #         gc.collect()

exit(0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    curr_date = arrow.get(
        datetime.datetime(2021, 8, 1, 0, 0, 0),
        ClockController.time_zone
    )  # 2021-07-22

    while True:
        start_time = curr_date.shift(hours=9).shift(minutes=30)
        end_time = curr_date.shift(hours=16).shift(minutes=00)
        ClockController.set_utcnow(start_time)

        robot_client = RobotClient(cliendId=0, simulator="influx")
        es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT')

        checkSignalId = robot_client.subscribe_bar_signal(
            es_key, BarSize.DAY_01, 100)
        checkSignal = robot_client.assetCache[es_key].signals[checkSignalId].get_numpy(
        )
        robot_client.unsubscribe_bar_signal(checkSignalId)
        time.sleep(0.1)

        max_days = checkSignal.shape[0]

        # set minimum number of trading days to allow
        if max_days < 30 or (curr_date - arrow.get(checkSignal[-1, 0])) > datetime.timedelta(days=1):
            # iterate day
            curr_date = curr_date.shift(days=1)
            robot_client.disconnect_client()
            continue

        bars = {
            BarSize.SEC_05: min(max_days*24*60*2, 300),
            BarSize.SEC_15: min(max_days*24*60*2, 300),
            BarSize.SEC_30: min(max_days*24*60*2, 300),
            BarSize.MIN_01: min(max_days*24*60, 250),
            BarSize.MIN_05: min(max_days*24*12, 200),
            BarSize.MIN_15: min(max_days*24*4, 150),
            BarSize.MIN_30: min(max_days*24*2, 120),
            BarSize.HRS_01: min(max_days*24, 90),
            BarSize.HRS_04: min(max_days*6, 60),
            BarSize.DAY_01: min(max_days, 30)
        }
        targetBars05 = {
            BarSize.SEC_05: 5//5,
            BarSize.SEC_15: 15//5,
            BarSize.SEC_30: 30//5,
            BarSize.MIN_01: 60//5,
            BarSize.MIN_05: 60//5*5,
            BarSize.MIN_10: 60//5*10,
            BarSize.MIN_15: 60//5*15
        }
        targetBars15 = {
            BarSize.MIN_30: 60//5*30//3,
            BarSize.HRS_01: 60//5*60//3
        }
        targetBars30 = {
            BarSize.HRS_02: 60//5*60*2//6
        }

        try:
            signalIds = {
                bar: robot_client.subscribe_bar_signal(es_key, bar, period)
                for bar, period in bars.items()
            }
        except:
            curr_date = curr_date.shift(days=1)
            robot_client.disconnect_client()
            continue

        dataWindow = {}
        curr_ts = ClockController.utcnow() .int_timestamp
        last_ts = end_time.int_timestamp + max(targetBars30.values())*30
        last_input_ts = end_time.int_timestamp
        while curr_ts < last_ts:
            time.sleep(0.001)
            if curr_ts < last_input_ts \
                    and curr_ts - 5 != robot_client.assetCache[es_key].signals[signalIds[BarSize.SEC_05]].get_numpy()[-1, 0]:
                dataWindow[curr_ts] = {
                    "inputData": {
                        bar.value: robot_client.assetCache[es_key].signals[signalIds[bar]].get_numpy(
                        ).tolist()
                        for bar in bars
                    },
                    "matchData": {}
                }

            for bar, lb in targetBars05.items():
                if (curr_ts - 5*lb) in dataWindow:
                    data = robot_client.assetCache[es_key].signals[signalIds[BarSize.SEC_05]].get_numpy(
                        lb)
                    dataWindow[(curr_ts - 5*lb)]["matchData"][bar.value] = {
                        'high': max(data.T[2]), 'low': min(data.T[3])}
            for bar, lb in targetBars15.items():
                if (curr_ts - 15*lb) in dataWindow:
                    data = robot_client.assetCache[es_key].signals[signalIds[BarSize.SEC_15]].get_numpy(
                        lb)
                    dataWindow[(curr_ts - 15*lb)]["matchData"][bar.value] = {
                        'high': max(data.T[2]), 'low': min(data.T[3])}
            for bar, lb in targetBars30.items():
                if (curr_ts - 30*lb) in dataWindow:
                    data = robot_client.assetCache[es_key].signals[signalIds[BarSize.SEC_30]].get_numpy(
                        lb)
                    dataWindow[(curr_ts - 30*lb)]["matchData"][bar.value] = {
                        'high': max(data.T[2]), 'low': min(data.T[3])}

            print(f"Done {ClockController.utcnow()}")
            ClockController.increment_utcnow(5)
            curr_ts = ClockController.utcnow().int_timestamp

        if len(dataWindow) > 5*60*60//5:
            contract_file = f"U:/MarketData/{es_key.replace('@', '_')}/HLS/{curr_date.format('YYYYMMDD')}.json"
                                                                                 with open(contract_file, 'w') as f:
                json.dump(dataWindow, f)
        gc.collect()

        for signalId in signalIds.values():
            robot_client.unsubscribe_bar_signal(signalId)

        # iterate day
        curr_date = curr_date.shift(days=1)
        robot_client.disconnect_client()
