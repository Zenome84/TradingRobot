
import math
from typing import List
import tensorflow as tf
import keras.api._v2.keras as krs

class ScaledDotProductAttention(krs.layers.Layer):
    def __init__(self,
        attention_size: int, attention_dropout=None, use_mask=True,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        if attention_dropout is None:
            attention_dropout = 0.
        self._dropout = krs.layers.Dropout(attention_dropout)
        self._softmax = krs.layers.Softmax() # krs.layers.Activation('softmax')
        self._attention = krs.layers.Lambda(lambda x: krs.backend.batch_dot(x[0], x[1], [2, 2]) / math.sqrt(attention_size))
        if use_mask:
            self._mask = krs.layers.Lambda(lambda x: (-1e+9) * (1. - krs.backend.cast(x, tf.float32)))
            self._add_mask = krs.layers.Add()
        self._match = krs.layers.Lambda(lambda x: krs.backend.batch_dot(x[0], x[1]))

    def call(self, query, key, value, mask=None, *args, **kwargs):

        scale = krs.backend.sqrt(krs.backend.cast(krs.backend.shape(key)[-1], tf.float32))
        attention_layer = self._attention([query, key])
        if mask is not None:
            mask_layer = self._mask(mask)
            masked_attention_layer = self._add_mask([attention_layer, mask_layer])
        else:
            masked_attention_layer = attention_layer
        
        masked_attention_layer_soft = self._softmax(masked_attention_layer)
        attention_dropout_layer = self._dropout(masked_attention_layer_soft)
        matched_layer = self._match(([attention_dropout_layer, value]))

        return matched_layer, attention_dropout_layer

class InterpretableMultiHeadAttention(krs.layers.Layer):
    def __init__(self,
        num_heads: int, model_size: int, use_mask=True, dropout_rate: float=None,
        trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._queries = [
            krs.layers.Dense(model_size, use_bias=False)
            for _ in range(num_heads)
        ]
        self._keys = [
            krs.layers.Dense(model_size, use_bias=False)
            for _ in range(num_heads)
        ]
        self._value = krs.layers.Dense(model_size, use_bias=False)
        
        self._head_dropouts = [
            krs.layers.Dropout(dropout_rate)
            for _ in range(num_heads)
        ]
        self._output_dropout = krs.layers.Dropout(dropout_rate)
        
        self._attention = ScaledDotProductAttention(model_size, dropout_rate, use_mask)
        self._weights = krs.layers.Dense(model_size, use_bias=False)

    def call(self, query, key, value, mask=None, *args, **kwargs):
        num_heads = len(self._queries)

        heads = []
        attentions = []
        for i in range(num_heads):
            head, attention = self._attention(
                self._queries[i](query),
                self._keys[i](key),
                self._value(value),
                mask
            )

            head_dropout = self._head_dropouts(head)

            heads.append(head_dropout)
            attentions.append(attention)

        heads_layer = krs.backend.stack(heads) # if num_heads > 1 else heads[0]
        attentions_layer = krs.backend.stack(attentions)

        heads_mean_layer = krs.backend.mean(heads_layer, axis=0) # if num_heads > 1 else head
        weights_layer = self._weights(heads_mean_layer)
        weights_dropout_layer = self._output_dropout(weights_layer)  # output dropout

        return weights_dropout_layer, attentions_layer
