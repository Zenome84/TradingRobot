
from itertools import accumulate
import math
from msilib.schema import Error
from typing import Dict, List
import arrow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats as st
import tensorflow_transform as tft
import keras.api._v2.keras as krs

from importlib import import_module
from resources.enums import BarColumn, BarDeltaSeconds

from resources.time_tools import ClockController

from ai.tft.TemporalFusionTransformer import TemporalFusionTransformer

class TimeseriesDatasetGenerator():
    @staticmethod
    def time_of_day(
        current_utc_time, start_time,
        modes = None, end_time = None, offset_time = None):
        '''
        Converts utc_time (can be array) into time_of_day (all times in utc format)
            modes is a list that can contain 'r' for relative and 'o' for offset, or both
                ommiting modes means not relative nor offset.
            if you want relative, then you must provide end_time
            if you want offset, then you must provide offset_time
        '''
        modes = [] if modes is None else modes
        if 'r' in modes:
            if 'o' in modes:
                return (current_utc_time + offset_time - start_time) / (end_time - start_time)
            else:
                return (current_utc_time - start_time) / (end_time - start_time)
        else:
            if 'o' in modes:
                return current_utc_time + offset_time - start_time
            else:
                return current_utc_time - start_time

    @staticmethod
    def make_input_sample(
        bar_data, valid_day: arrow.Arrow, forecast_size,
        tod_modes = None, bar_delta: int = None):
        '''
        '''
        day_of_week = valid_day.weekday()
        bod_ts = valid_day.replace(hour=9, minute=30).int_timestamp
        eod_ts = valid_day.replace(hour=16).int_timestamp

        bar_data = np.array(bar_data)
        timestamp_ts = bar_data[:, BarColumn.TimeStamp.value]
        volume_ts = bar_data[:, BarColumn.Volume.value]
        price_ts = bar_data[:, [
            BarColumn.Open.value,
            BarColumn.High.value,
            BarColumn.Low.value,
            BarColumn.Close.value,
            BarColumn.VWAP.value
        ]]

        time_of_day_ts = TimeseriesDatasetGenerator.time_of_day(
            timestamp_ts, bod_ts, tod_modes, eod_ts, bar_delta)
        
        timestamp_ts_forecast: np.ndarray = np.arange(
            timestamp_ts[-1] + bar_delta,
            min(
                eod_ts + int('o' in tod_modes),
                timestamp_ts[-1] + forecast_size*bar_delta
            ),
            bar_delta
        )
        time_of_day_ts_forecast = TimeseriesDatasetGenerator.time_of_day(
            timestamp_ts_forecast, bod_ts, tod_modes, eod_ts, bar_delta)

        return {
            'static': {
                'day_of_week': day_of_week
            },
            'observed': {
                'time_of_day_ts': time_of_day_ts,
                'volume_ts': volume_ts,
                'price_ts': price_ts
            },
            'forecast': {
                'time_of_day_ts': time_of_day_ts_forecast
            }
        }

    def __init__(self,
        all_data, window_sizes, forecast_sizes, tvt_split, batch_size,
        tod_modes=None):
        '''
        Structure of all_data:
            {
                day_ts: {
                    bar_size: bar_timeseries,
                    ...
                },
                ...
            }
        
        Structure of window_sizes/forecast_sizes:
            {
                bar_size: size
            }

        Structure of tvt_split:
            [% Train Cutoff, % Validate Cutoff]
            Test will be anything after validate cutoff.
            To have only Train use: [1, 1]
            To have only Train/Validate use: [0.X, 1]

        Input batch_size must be a positive integer
        
        See TimeseriesDatasetGenerator.time_of_day documentation for tod_modes
        '''
        if (len(tvt_split) != 2) or not(0 < tvt_split[0] <= tvt_split[1] <= 1):
            raise NotImplementedError(f"TimeseriesDatasetGenerator: tvt_split must be length 2, with 0 < tvt_split[0] <= tvt_split[1] <= 1. Got: {tvt_split}")
        if batch_size <= 0:
            raise NotImplementedError(f"TimeseriesDatasetGenerator: batch_size must be greater than 0. Got: {batch_size}")

        self.total_num_days = len(all_data)

        train_days_end = math.floor(tvt_split[0]*self.total_num_days)
        validate_days_end = math.floor(tvt_split[1]*self.total_num_days)

        train_data: Dict[str, List[dict]] = dict()
        validate_data: Dict[str, List[dict]] = dict()
        test_data: Dict[str, List[dict]] = dict()
        for bar_len in window_sizes.keys():
            train_data[bar_len] = list()
            validate_data[bar_len] = list()
            test_data[bar_len] = list()

        day_iterator = 0
        model = TemporalFusionTransformer(None, None, 5, 2, 0)
        for valid_day, day_data in all_data.items():
            valid_day_arrow = arrow.get(int(valid_day), tzinfo=ClockController.time_zone)
            day_of_week = valid_day_arrow.weekday()
            bod_ts = valid_day_arrow.replace(hour=9, minute=30).int_timestamp
            eod_ts = valid_day_arrow.replace(hour=16).int_timestamp

            for bar_len, bar_data in day_data.items():
                wnd_size = window_sizes[bar_len]
                frc_size = forecast_sizes[bar_len]

                bar_data = np.array(bar_data)
                timestamp_ts = bar_data[:, BarColumn.TimeStamp.value]
                volume_ts = bar_data[:, BarColumn.Volume.value]
                price_ts = bar_data[:, [
                    BarColumn.Open.value,
                    BarColumn.High.value,
                    BarColumn.Low.value,
                    BarColumn.Close.value,
                    BarColumn.VWAP.value
                ]]

                bar_delta = BarDeltaSeconds[bar_len]
                time_of_day_ts = TimeseriesDatasetGenerator.time_of_day(
                    timestamp_ts, bod_ts, tod_modes, eod_ts, bar_delta)
                
                for k in range(bar_data.shape[0] - wnd_size):
                    wnd_k = k + wnd_size
                    time_of_day_ts_observed = time_of_day_ts[k : wnd_k]
                    volume_ts_observed = volume_ts[k : wnd_k]
                    price_ts_observed = price_ts[k : wnd_k]

                    time_of_day_ts_forecast = time_of_day_ts[wnd_k: wnd_k + frc_size]
                    high_ts_target = price_ts[wnd_k: wnd_k + frc_size, 1]
                    low_ts_target = price_ts[wnd_k: wnd_k + frc_size, 2]

                    sample = {
                        'inputs': {
                            'static': {
                                'day_of_week': day_of_week
                            },
                            'observed': {
                                'time_of_day_ts': time_of_day_ts_observed,
                                'volume_ts': volume_ts_observed,
                                'price_ts': price_ts_observed
                            },
                            'forecast': {
                                'time_of_day_ts': time_of_day_ts_forecast
                            }
                        },
                        'targets': {
                            'high_ts': high_ts_target,
                            'low_ts': low_ts_target
                        }
                    }
                    
                    static_inputs = np.reshape(day_of_week, (1, 1))
                    observed_inputs = np.concatenate([
                        np.reshape(time_of_day_ts_observed, (1, -1, 1)),
                        np.reshape(volume_ts_observed, (1, -1, 1)),
                        np.reshape(price_ts_observed, (1, -1, 5))
                    ], axis=-1)
                    forecast_inputs = np.reshape(time_of_day_ts_forecast, (1, -1, 1))

                    pred =  model.predict((static_inputs, observed_inputs, forecast_inputs))

                    if day_iterator < train_days_end:
                        train_data[bar_len].append(sample)
                    elif day_iterator < validate_days_end:
                        validate_data[bar_len].append(sample)
                    else:
                        test_data[bar_len].append(sample)
                
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data