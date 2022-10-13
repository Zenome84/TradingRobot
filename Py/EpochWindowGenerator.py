

from itertools import accumulate
import math
from msilib.schema import Error
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as krs
import scipy.stats as st
import tensorflow_transform as tft


class EpochWindowGenerator():
    def __init__(self, data, input_len, label_len, shift, train_cut, validate_cut, batch_size):
        if len(data.shape) != 3:
            raise NotImplementedError(f"EpochWindowGenerator data must be of rank 3 (Epoch, Time, Feature), but found rank {len(data.shape)}")

        self.num_epochs, self.epoch_size, self.num_features = data.shape

        self.input_len = input_len
        self.label_len = label_len
        self.shift = shift
        self.total_window_size = input_len + shift
        

        self.input_slice = slice(0, self.input_len)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_len
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._train_ds = None
        self._validate_ds = None
        self._test_ds = None
        self.batch_size = batch_size

        for e in range(self.num_epochs):
            if e < train_cut*self.num_epochs:
                self._train_ds = self.make_dataset(data[e], self._train_ds)
            elif e < validate_cut*self.num_epochs:
                self._validate_ds = self.make_dataset(data[e], self._validate_ds)
            else:
                self._test_ds = self.make_dataset(data[e], self._test_ds)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]

        inputs.set_shape([self.input_len, None])
        labels.set_shape([self.label_len, None])

        return inputs, labels

    def plot(self, model=None, plot_col_index=None, max_subplots=3, from_ds=None, style=None):
        if plot_col_index is None:
            plot_col_index = 0
        plt.figure(figsize=(12, 8))
        if from_ds in {None, 'test'}:
            inputs, labels = next(iter(self._test_ds.shuffle(len(self._test_ds)).batch(max_subplots)))
        elif from_ds == 'train':
            inputs, labels = next(iter(self._test_ds.shuffle(len(self._train_ds)).batch(max_subplots)))
        elif from_ds == 'validate':
            inputs, labels = next(iter(self._test_ds.shuffle(len(self._validate_ds)).batch(max_subplots)))
        else:
            raise ValueError(f"Expected test, train, or validate for 'from_ds' but got: {from_ds}")
        for n in range(max_subplots):
            plt.subplot(max_subplots, 1, n+1)
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            plt.scatter(self.label_indices, labels[n, :, plot_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions, mu, sigma = model(inputs[n:n+1])
                plt.scatter(self.label_indices, EpochWindowGenerator.accumulate_histogram(predictions[0], mu[0], sigma[0], style=style)[:, plot_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data, prev_data=None):
        data = np.array(data, dtype=np.float32)
        ds = krs.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                # shuffle=True,
                batch_size=None,
            )

        ds = ds.map(self.split_window)

        if prev_data is None:
            return ds
        else:
            return prev_data.concatenate(ds)
    
    @staticmethod
    def make_histogram(input_tensor, mu=0., sigma=1.25, df=15, bins=32):
        boundaries = np.array([st.t.ppf(np.linspace(0.0001, 0.9999, bins-1), df)])*sigma + mu
        return tf.one_hot(tft.apply_buckets(input_tensor, boundaries), bins, axis=-1)

    @staticmethod
    def accumulate_histogram(input_tensor, mu=0., sigma=1.25, df=15, bins=32, style=None):
        boundaries = np.array([st.t.ppf(np.linspace(0.0001, 0.9999, bins-1), df)])*sigma + mu
        bucket_values = tf.constant(
            np.concatenate([
                boundaries[:,0:1] - (boundaries[:,1:2] - boundaries[:,0:1])*0.75,
                (boundaries[:,:-1] + boundaries[:,1:])/2,
                boundaries[:,-1:] + (boundaries[:,-1:] - boundaries[:,-2:-1])*0.75
            ], -1), dtype=tf.float32
        )

        if style in {None, 'mode'}:
            input_tensor = tf.one_hot(tf.math.argmax(input_tensor, -1), bins, axis=-1)
            return tf.reduce_sum(input_tensor * bucket_values, -1)
        if style == 'mean':
            return tf.reduce_sum(input_tensor * bucket_values, -1)

    @property
    def train_ds(self):
        return self._train_ds.shuffle(buffer_size=len(self._train_ds), reshuffle_each_iteration=True).batch(self.batch_size)

    @property
    def validate_ds(self):
        return self._validate_ds.batch(self.batch_size)

    @property
    def test_ds(self):
        return self._test_ds.batch(self.batch_size)