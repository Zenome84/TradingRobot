
import datetime
import json
import os
import arrow
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from ai.NormalizeStates import NormalizeStates
from ai.tft.TimeseriesDatasetGenerator import TimeseriesDatasetGenerator
from resources.enums import BarColumn, BarSize
from resources.time_tools import ClockController, wait_until

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
            'input_transform': lambda fname, data: data[:25, BarColumn.TimeStamp.value]
        },
        'volume_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.Volume.value]
        },
        'open_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.Open.value]
        },
        'high_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.High.value]
        },
        'low_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.Low.value]
        },
        'close_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.Close.value]
        },
        'vwap_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[:25, BarColumn.VWAP.value]
        }
    },
    'forecast': {
        'time_of_day_ts': {
            'num_categories': 0,
            'input_transform': lambda fname, data: data[25:, BarColumn.TimeStamp.value]
        }
    }
}
target_spec = {
    'high_ts': {
        'num_categories': 0,
        'quantiles': [0.1, 0.25, 0.5],
        'input_transform': lambda fname, data: data[25:, BarColumn.High.value]
    },
    'low_ts': {
        'num_categories': 0,
        'quantiles': [0.5, 0.75, 0.9],
        'input_transform': lambda fname, data: data[25:, BarColumn.Low.value]
    }
}

def get_relative_tod(time_series, begin_day, end_day, offset):
    return (time_series + offset - begin_day) / (end_day - begin_day)

# def get_weekday(timeseries):
#     np.apply_along_axis(
#         lambda x: arrow.get(x, tzinfo=ClockController.time_zone).weekday(),
#         0, timeseries
#     )

if __name__ == "__main__":
    
    # with open('data/allData.json', 'r') as fread:
    #     allData = json.load(fread)
    
    # for ts, ts_data in allData.items():
    #     # for bar_len, bar_data in ts_data.items():
    #     #     # day_path = f"data/{bar_len.replace(' ', '_')}/{ts}.json"
    #     #     # with open(day_path, 'w') as fwrite:
    #     #     #     json.dump(bar_data, fwrite)
    #     #     day_path = f"data/{bar_len.replace(' ', '_')}/{ts}.csv"
    #     #     np.savetxt(day_path, bar_data, delimiter=',')
        
    #     valid_day = arrow.get(int(ts), tzinfo=ClockController.time_zone)
    #     day_of_week = valid_day.weekday()
    #     bod_ts = valid_day.replace(hour=9, minute=30).int_timestamp
    #     eod_ts = valid_day.replace(hour=16).int_timestamp
        
    #     with open('data/day_metadata/{ts}.json', 'w') as fw:
    #         json.dump({
    #             'day_of_week': day_of_week,
    #             'bod_ts': bod_ts,
    #             'eod_ts': eod_ts
    #         }, fw)

    # exit()
    # dow = get_weekday(tf.constant(1545195600))

    # print(dow)

    # files_n_data = {
    #     '0': tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
    #     '1': tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]) + 1,
    #     '2': tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]) + 2
    # }

    # def gen_data():
    #     for key, val in files_n_data.items():
    #         yield tf.strings.to_number(key, tf.int32), tf.data.Dataset.from_tensor_slices(val)

    # ds = tf.data.Dataset.from_generator(gen_data, output_signature=(
    #     tf.TensorSpec(shape=(), dtype=tf.int32),
    #     tf.data.DatasetSpec(tf.TensorSpec(shape=(4), dtype=tf.int32))
    # ))

    # def flatten_data(dataset):
    #     for num, sub_ds in dataset:
    #         for data in sub_ds:
    #             yield tf.concat([[num], data], 0)

    # ds2 = tf.data.Dataset.from_generator(lambda: flatten_data(ds), output_signature=tf.TensorSpec(shape=(5), dtype=tf.int32))

    # for data in ds2:
    #     print(data)

    # for num, sub_ds in ds:
    #     print(num)
    #     for data in sub_ds:
    #         print(data)

    data_path = 'data/15_mins'

    # @tf.function
    def csv_to_tensor(fname):
        NUMERIC_TYPE = tf.float64

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
        ts = to_tensor_series[-1, 0] + tf.range(time_delta, time_delta*(5+1), time_delta)
        price = to_tensor_series[-1, 4] * tf.ones([5, 2], dtype=NUMERIC_TYPE)
        
        repeated_tensor = tf.concat([
            tf.expand_dims(ts, -1),
            tf.zeros([5, 1], dtype=NUMERIC_TYPE),
            price,
            tf.zeros([5, 4], dtype=NUMERIC_TYPE)
        ], -1)

        return meta, tf.concat([to_tensor_series, repeated_tensor], 0)
    def sub_to_batch(sub):
        return sub.batch(105, drop_remainder=True)

    pattern = "1545195600"
    # pattern = "*"
    dataset = tf.data.Dataset.list_files(f"{data_path}/{pattern}.csv", shuffle=False)
    dataset = dataset.map(csv_to_tensor)
    dataset = dataset.map(
        lambda m, t: 
        (m, tf.data.Dataset.from_tensor_slices(t).window(105, 1).flat_map(sub_to_batch))
    )

    def ds_to_spec(dataset):
        for meta, window_data in dataset:
            for data in window_data:
                inputs=(
                    meta["day_of_week"],
                    data[:100,0]-meta["bod_ts"],
                    data[:100,5],
                    data[:100,1],
                    data[:100,2],
                    data[:100,3],
                    data[:100,4],
                    data[:100,7],
                    data[100:,0]-meta["bod_ts"]
                )
                outputs=(
                    data[100:,2],
                    data[100:,3]
                )
                yield inputs, outputs
    
    dataset2 = tf.data.Dataset.from_generator(
        lambda: ds_to_spec(dataset),
        output_signature=(
            (
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[100], dtype=tf.float64),
                tf.TensorSpec(shape=[5], dtype=tf.float64)
            ),
            (
                tf.TensorSpec(shape=[5], dtype=tf.float64),
                tf.TensorSpec(shape=[5], dtype=tf.float64)
            )
        )
    )

    for data in dataset2:
        print(data)

    for day_ts, day_dataset in dataset:
        for window_data in day_dataset:
            # print(window_data[:,0])
            print(window_data[-1])


    for sub_dataset in dataset:
        def sub_to_batch(sub):
            return sub.batch(30, drop_remainder=True)
        day_dataset = sub_dataset.window(30, 1)
        day_dataset = day_dataset.flat_map(sub_to_batch)
        for window_data in day_dataset:
            print(window_data[:,0])
    
    with open('data/allData.json', 'r') as fread:
        allData = json.load(fread)
    
    for ts, ts_data in allData.items():
        for bar_len, bar_data in ts_data.items():
            # day_path = f"data/{bar_len.replace(' ', '_')}/{ts}.json"
            # with open(day_path, 'w') as fwrite:
            #     json.dump(bar_data, fwrite)
            day_path = f"data/{bar_len.replace(' ', '_')}/{ts}.csv"
            np.savetxt(day_path, bar_data, delimiter=',')

    window_sizes = {
        BarSize.SEC_01.value: 600,
        BarSize.MIN_01.value: 100,
        BarSize.MIN_05.value: 100,
        BarSize.MIN_15.value: 100,
        BarSize.MIN_30.value: 100,
    }
    forecast_sizes = {
        BarSize.SEC_01.value: 90,
        BarSize.MIN_01.value: 15,
        BarSize.MIN_05.value: 12,
        BarSize.MIN_15.value: 6,
        BarSize.MIN_30.value: 4,
    }
    test = TimeseriesDatasetGenerator(allData, window_sizes, forecast_sizes, [1,1], 5, ['r','o'])

    valid_day = arrow.get(int(next(iter(allData.keys()))), tzinfo=ClockController.time_zone)
    bod_ts = arrow.get(datetime.datetime(
        valid_day.year, valid_day.month, valid_day.day, 9, 30, 0), ClockController.time_zone).int_timestamp
    eod_ts = arrow.get(datetime.datetime(
        valid_day.year, valid_day.month, valid_day.day, 16, 00, 0), ClockController.time_zone).int_timestamp
    
    sec_in_trade_day = bod_ts - eod_ts
    day_of_week = arrow.get(valid_day, tzinfo=ClockController.time_zone).weekday()

    val_dict = next(iter(allData.values()))
    sec01 = np.array(val_dict['sec01'])
    sec01[:, 0] = sec01[:, 0]



    # ns_layer = NormalizeStates()
    # out_layer = ns_layer(np.array([sec01]), np.array([bod_ts]), np.array([eod_ts]))

    exit()



    # @staticmethod
    # def read_file(fpath):
    #     fpath_split = tf.strings.split(fpath, os.path.sep)
        
    #     ts = tf.strings.to_number(tf.strings.split(fpath_split[-1], '.')[0], out_type=NUMERIC_TYPE)
    #     am_pm = tf.strings.split(fpath_split[-2], '.')
    #     am_ts = tf.strings.to_number(am_pm[0], out_type=NUMERIC_TYPE)
    #     pm_ts = tf.strings.to_number(am_pm[1], out_type=NUMERIC_TYPE)

    #     return (tfio.experimental.serialization.decode_json(tf.io.read_file(fpath), specs=FILE_SPECS), (ts - am_ts) / (pm_ts - am_ts))

    # @staticmethod
    # def verify_file(json_obj, ts_obj):
    #     verify = tf.concat([
    #         [tf.cast(tf.not_equal(tf.shape(json_obj["inputData"][bar.value]), tf.TensorShape([bar_length, 8])), tf.int32)]
    #         for bar, bar_length in INPUT_DATA_SPECS.items()
    #     ], axis = 0)
    #     return tf.equal(0, tf.reduce_sum(verify))

    # @staticmethod
    # def process_file(json_obj, ts_obj):
    #     dict_obj = json_obj
    #     dict_obj["inputData"]["timeOfDay"] = ts_obj
    #     dict_obj["inputData"]["currPrice"] = json_obj["inputData"][BarSize.SEC_05.value][-1][4]

    #     return dict_obj

    # @staticmethod
    # def preprocess_dict(dict_obj):
    #     currPrice = dict_obj["inputData"]["currPrice"]
    #     tod = dict_obj["inputData"]["timeOfDay"]
        
    #     inputData = tf.concat([
    #         tf.concat([
    #             dict_obj["inputData"][bar.value][:, 1:5]/currPrice,
    #             (dict_obj["inputData"][bar.value][:, 5:6] - tf.reduce_mean(dict_obj["inputData"][bar.value][:, 5:6])) \
    #                 / tf.math.reduce_std(dict_obj["inputData"][bar.value][:, 5:6])
    #         ], axis=1)
    #         for bar in INPUT_DATA_SPECS.keys()
    #     ], axis=0)
    #     inputData = tf.concat([inputData, [[tod, currPrice, 0, 0, 0]]], axis=0)

    #     buckets = np.array([range(63)])/63*0.012
    #     matchData = tf.transpose(tf.concat([
    #         [[val["high"], val["low"]]]
    #         for val in dict_obj["matchData"].values()
    #     ], axis=0))
    #     # matchData = tf.transpose(tf.concat([
    #     #     [tf_accumulate(matchData[0], 'max')/currPrice], 
    #     #     [tf_accumulate(matchData[1], 'min')/currPrice]
    #     # ], axis=0))
    #     matchData = tf.transpose(tf.concat([
    #         [tft.apply_buckets(
    #             tf_accumulate(matchData[0], 'max')/currPrice,
    #             1 + buckets
    #         )], 
    #         [63 - tft.apply_buckets(
    #             tf_accumulate(matchData[1], 'min')/currPrice,
    #             np.flip(1 - buckets)
    #         )]
    #     ], axis=0))
    #     matchData = tf.one_hot(matchData, 64)
    #     return (inputData, matchData)

