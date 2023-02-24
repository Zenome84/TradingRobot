
from enum import Enum
from typing import Tuple
import tensorflow as tf
import keras.api._v2.keras as krs

class LongShort(Enum):
    """"""
    LONG = 'Long'
    SHORT = 'Short'

class NormalizeState(krs.layers.Layer):
    """"""
    def __init__(self,
        trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def call(self,
        current_time: tf.Tensor,
        begin_end_trading_time: tf.Tensor,
        time_timeseries: tf.Tensor,
        price_timeseries: tf.Tensor,
        volume_timeseries: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """"""
        # price_timeseries_shape = tf.shape(price_timeseries)
        # volume_timeseries_shape = tf.shape(volume_timeseries)

        current_price = price_timeseries[:, 0, -1, self._price_idx]
        current_volume = volume_timeseries[:, :, -1]

        # flat_price_timeseries_shape = tf.constant([price_timeseries_shape[0], tf.reduce_prod(price_timeseries_shape[1:])])
        logprice_timeseries = tf.math.log(price_timeseries/current_price)

        # flat_volume_timeseries_shape = tf.constant([volume_timeseries_shape[0], tf.reduce_prod(volume_timeseries_shape[1:])])
        norm_volume_timeseries = volume_timeseries/current_volume

        bod_time = begin_end_trading_time[0]
        eod_time = begin_end_trading_time[1]
        relative_time_timeseries = (time_timeseries - bod_time) / (eod_time - bod_time)
        relative_current_time = (current_time - bod_time) / (eod_time - bod_time)

        return (relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries)

class Critic(krs.layers.Layer):
    """"""
    def __init__(self,
        long_short: LongShort,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._long_short = long_short

    def call(self,
        current_position: tf.Tensor,
        relative_current_time: tf.Tensor,
        current_volume: tf.Tensor,
        relative_time_timeseries: tf.Tensor,
        logprice_timeseries: tf.Tensor,
        norm_volume_timeseries: tf.Tensor, training=False
        ) -> tf.Tensor:
        ...

class Actor(krs.layers.Layer):
    """"""
    def __init__(self,
        long_short: LongShort,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._long_short = long_short

    def call(self,
        current_position: tf.Tensor,
        relative_current_time: tf.Tensor,
        current_volume: tf.Tensor,
        relative_time_timeseries: tf.Tensor,
        logprice_timeseries: tf.Tensor,
        norm_volume_timeseries: tf.Tensor, training=False
        ) -> tf.Tensor:
        ...

class ActorCritic(krs.layers.Layer):
    """"""
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._normalize_state = NormalizeState()
        self._long_actor
        self._short_actor
        self._long_critic
        self._short_critic

    def call(self,
        current_position: tf.Tensor,
        current_time: tf.Tensor,
        begin_end_trading_time: tf.Tensor,
        time_timeseries: tf.Tensor,
        price_timeseries: tf.Tensor,
        volume_timeseries: tf.Tensor):
        
        relative_current_time, \
            current_price, \
            current_volume, \
            relative_time_timeseries, \
            logprice_timeseries, \
            norm_volume_timeseries = self._normalize_state(
                current_time,
                begin_end_trading_time,
                time_timeseries,
                price_timeseries,
                volume_timeseries
            )

        long_order = self._long_actor(
            current_position,
            relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries
        )

        short_order = self._short_actor(
            current_position,
            relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries
        )
        
        long_valuation = self._long_critic(
            current_position,
            relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries,
            long_order,
            short_order
        )
        
        short_valuation = self._short_critic(
            current_position,
            relative_current_time,
            current_price,
            current_volume,
            relative_time_timeseries,
            logprice_timeseries,
            norm_volume_timeseries,
            long_order,
            short_order
        )