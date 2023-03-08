
import datetime
import json
import arrow
import numpy as np
from ai.NormalizeStates import NormalizeStates
from ai.tft.TimeseriesDatasetGenerator import TimeseriesDatasetGenerator
from resources.enums import BarSize
from resources.time_tools import ClockController, wait_until


def get_relative_tod(time_series, begin_day, end_day, offset):
    return (time_series + offset - begin_day) / (end_day - begin_day)

def get_weekday(timeseries):
    np.apply_along_axis(
        lambda x: arrow.get(x, tzinfo=ClockController.time_zone).weekday(),
        0, timeseries
    )

if __name__ == "__main__":
    
    with open('data/allData.json', 'r') as fread:
        allData = json.load(fread)
    
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

