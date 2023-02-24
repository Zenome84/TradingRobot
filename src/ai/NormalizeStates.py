
from enum import Enum
from typing import Tuple
import tensorflow as tf
import keras.api._v2.keras as krs
from resources.enums import BarColumn

class NormalizeStates(krs.layers.Layer):
    """"""
    def __init__(self,
        trainable=None, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def call(self,
        bar_timeseries: tf.Tensor, begin_trading_time: tf.Tensor, end_trading_time: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """"""
        ts_len = bar_timeseries.shape[-2]
        time_timeseries = tf.gather(bar_timeseries, [
            BarColumn.TimeStamp.value
        ], axis=-1)
        volume_timeseries = tf.gather(bar_timeseries, [
            BarColumn.Volume.value
        ], axis=-1)
        price_timeseries = tf.gather(bar_timeseries, [
            BarColumn.Open.value,
            BarColumn.High.value,
            BarColumn.Low.value,
            BarColumn.Close.value,
            BarColumn.VWAP.value
        ], axis=-1)

        current_time = tf.gather(time_timeseries, [ts_len-1], axis=-2)
        current_volume = tf.gather(volume_timeseries, [ts_len-1], axis=-2)
        current_price = tf.gather(tf.gather(price_timeseries, [ts_len-1], axis=-2), [3], axis=-1)

        # flat_price_timeseries_shape = tf.constant([price_timeseries_shape[0], tf.reduce_prod(price_timeseries_shape[1:])])
        logprice_timeseries = tf.math.log(price_timeseries/current_price)

        # flat_volume_timeseries_shape = tf.constant([volume_timeseries_shape[0], tf.reduce_prod(volume_timeseries_shape[1:])])
        norm_volume_timeseries = volume_timeseries/current_volume

        relative_time_timeseries = (time_timeseries - begin_trading_time) / (end_trading_time - begin_trading_time)
        relative_current_time = (current_time - begin_trading_time) / (end_trading_time - begin_trading_time)

        return (relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries)
