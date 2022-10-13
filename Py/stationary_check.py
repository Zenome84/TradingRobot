
from tqdm import tqdm
import time
import arrow
import datetime
import numpy as np
from core.robot_client import RobotClient
from resources.enums import BarSize
from resources.time_tools import ClockController, wait_until
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as krs
import pandas as pd
from scipy import signal
import pywt
from EpochWindowGenerator import EpochWindowGenerator
import scipy.stats as st
import tensorflow_transform as tft


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                df, train_cut, val_cut,
                label_columns=None):
        # Store the raw data.
        n = len(df)
        self.train_df = df[0:int(n*train_cut)]
        self.val_df = df[int(n*train_cut):int(n*val_cut)]
        self.test_df = df[int(n*val_cut):]
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col=None, max_subplots=3):
        if plot_col is None:
            plot_col = self.label_columns[0]
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if not self.label_columns.empty:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = krs.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

class NormedLSTM(krs.Model):
    def __init__(self, units):
        super().__init__()
        self.lstm = krs.layers.LSTM(units, return_sequences=True)
        self.dense = krs.layers.Dense(units=1, activation='tanh', kernel_initializer=tf.initializers.zeros())

    def call(self, input_tensor):
        normalizer_mean = tf.math.reduce_mean(input_tensor, 1, True)
        normalizer_std = 5*tf.math.reduce_std(input_tensor, 1, True)
        normed_tensor = (input_tensor - normalizer_mean) / normalizer_std
        lstm_tensor = self.lstm(normed_tensor)
        dense_tensor = self.dense(lstm_tensor)
        output_tensor = (dense_tensor * normalizer_std) + normalizer_mean
        return output_tensor


def make_histogram(input_tensor, mu=0., sigma=1.25, df=15, bins=32):
    boundaries = np.array([st.t.ppf(np.linspace(0.0001, 0.9999, bins-1), df)])*sigma + mu
    return tf.one_hot(tf.map_fn(lambda x: tft.apply_buckets(x[0], x[1]), (input_tensor, boundaries), fn_output_signature=tf.int64), bins, axis=-1)
    

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

class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_feat):
        super().__init__()
        self.num_feat = num_feat
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_feat, activation='softmax')

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        normalizer_mean = tf.math.reduce_mean(inputs, 1, True)
        normalizer_std = tf.math.reduce_std(inputs, 1, True)
        inputs = make_histogram(inputs, normalizer_mean, normalizer_std)
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs[:,:,0,:])

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.expand_dims(tf.transpose(predictions, [1, 0, 2]), -2)
        # predictions = predictions * normalizer_std + normalizer_mean
        return predictions, normalizer_mean, normalizer_std
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred, mu, sigma = self(x, training=True)  # Forward pass
            y_true = make_histogram(y, mu, sigma)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred, mu, sigma = self(x, training=False)
        y_true = make_histogram(y, mu, sigma)
        # Updates the metrics tracking the loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def compile_and_fit(model, window, max_epochs=100, patience=2):
    early_stopping = krs.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

    # class CustomLoss(krs.losses.Loss):
    #     def call(self, y_true, y_pred):
    #         cat_loss = tf.keras.losses.CategoricalCrossentropy()
    #         y_predict, mu, sigma = y_pred
    #         y_label = make_histogram(y_true, mu, sigma)
    #         return cat_loss(y_label, y_predict)


    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=krs.optimizers.Adam(),
                    metrics=[krs.metrics.MeanAbsoluteError()])

    history = model.fit(window.train_ds, epochs=max_epochs,
                        validation_data=window.validate_ds,
                        callbacks=[early_stopping]
                        )
    return history


def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff(series, d, threshold=1e-2):
    slen = len(series)
    w = getWeights(d, slen)
    w_ = np.cumsum(abs(w))
    w_ /= w[-1]
    skip = w_[w_ > threshold].shape[0]
    df = []
    for k in range(skip, slen):
        df.append(np.dot(w[-k-1:, :].T, series[:k+1]))
    df = np.array(df)
    return df[:, 0]

def getWeights_FFD(d, threshold):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d-k+1)
        if abs(w_) < threshold: break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def fracDiff_FFD(series, d, threshold=1e-5):
    slen = len(series)
    w = getWeights_FFD(d, threshold)
    width = len(w)-1
    df = []
    for k in range(width, slen):
        df.append(np.dot(w.T, series[k-width:k+1]))
    return np.array(df)[:, 0]

# for d in np.arange(0.38, 0.41, 0.005):
#   df = fracDiff(np.log(vwap), d, 1.)[800:, 0]
#   df_wnd = np.lib.stride_tricks.sliding_window_view(df, 60)
#   u, s, vh = np.linalg.svd(df_wnd)
#   plt.plot(s/sum(s), label=f"{d}")
# plt.legend()
# plt.show()

# tests = []
# for d in np.arange(0., 0.25, 0.01):
#   df = fracDiff_FFD(vwap, d)[:, 0]
#   corr = np.corrcoef(vwap[-len(df):], df)
#   adf = adfuller(df, maxlag=1, regression='c', autolag=None)
#   tests.append([d, adf[1], corr[0, 1]])
# tests = np.array(tests)
# plt.plot(tests[:, 0], tests[:, 1:])
# plt.hlines(0.01, 0, 0.25)

# for d in np.arange(0., 1.1, 0.1):
#   df = fracDiff_FFD((vwap), d)
#   plt.plot(np.arange(len(vwap)-len(df), len(vwap)), df, label=f"{d}")
# plt.legend()
# plt.show()




if __name__ == "__main__":

    
    # # robot_client = RobotClient(cliendId=0, live=False)
    # robot_client = RobotClient(cliendId=0, simulator="influx")
    # # es_key = robot_client.subscribe_asset('SPY', 'SMART', 'STK')

    # # from resources.ibapi_orders import Orders
    # # orders = Orders.BracketOrder(10, "BUY", 1, 4579.25, 4580.5, 4578.0)
    # # for order in orders:
    # #     robot_client.client_adapter.placeOrder(order.orderId, robot_client.assetCache[es_key].contract, order)
    
    
    # data = []
    # ClockController.set_utcnow(arrow.get(datetime.datetime(
    #     2020, 7, 14, 9, 30, 0), ClockController.time_zone))
    # utc_now = ClockController.utcnow()
    # for _ in range(300):
    #     es_key = robot_client.subscribe_asset('ES', 'GLOBEX', 'FUT')
    #     wait_until(
    #         condition_function=lambda: robot_client.assetCache[es_key].updateOrderRulesObtained,
    #         seconds_to_wait=1,
    #         msg=f"Waited more than 1 secs to update order rules."
    #     )
    #     print(f"Asset: {es_key} | Commission: {robot_client.assetCache[es_key].commission} | Initial Margin: {robot_client.assetCache[es_key].initMargin} | Maintenance Marging: {robot_client.assetCache[es_key].maintMargin}")
    #     try:
    #         reqId = robot_client.subscribe_bar_signal(es_key, BarSize.MIN_01, 360)
    #         for _ in tqdm(range(360)):
    #             ClockController.increment_utcnow(60)
    #             time.sleep(0.001)
    #         data.append(robot_client.assetCache[es_key].signals[reqId].get_numpy())
    #         robot_client.unsubscribe_asset('ES', 'GLOBEX')
    #     finally:
    #         utc_now = utc_now.shift(days=1)
    #         while utc_now.isoweekday() not in [1, 2, 3, 4, 5]:
    #             utc_now = utc_now.shift(days=1)
    #         ClockController.set_utcnow(utc_now)

    # data = np.array(data)

    # # # plt.plot(data[:, 0], data[:, 2:4])
    # # # plt.plot(data[:, 0], data[:, 7])
    # # # plt.plot(data[-1, 0], data[-1, 4], marker='o')
    # # # plt.show()
    # # vwap = data
    # np.save("./Py/data.npy", data)

    vwap = np.load("./Py/data.npy")[:, :, 7:8]
    diffs = np.diff(vwap, 1, axis=1)
    print(f"VWAP Shape: {vwap.shape} | Diff Shape: {diffs.shape}")
    wnd = EpochWindowGenerator(vwap, 60, 15, 15, 0.7, 0.9, 128)
    wnd.plot(plot_col_index=0)

    # start = 900
    # rng = range(start, start+390)
    # rng = range(data.shape[0])
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(data[rng, 0], data[rng, 2:4]-data[rng, 7:])
    # axs[1].plot(data[rng, 0], data[rng, 7])
    # fig.show()

    # df = pd.DataFrame(vwap[: ,2:5], columns=['High', 'Low', 'Close'])
    # df = fracDiff_FFD(vwap[:, 2:5], 0.4)
    # df = pd.DataFrame(np.concatenate([vwap[-df.shape[0]:, 2:5], df], 1), columns=['High', 'Low', 'Close', 'dHigh', 'dLow', 'dClose'])
    # wnd = WindowGenerator(input_width=60, label_width=15, shift=15, df=df, train_cut=0.7, val_cut=0.9,
    #                  label_columns=df.columns)
    # wnd.plot(plot_col='Close')

    # lstm_model = krs.models.Sequential([
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     NormedLSTM(8),
    #     # Shape => [batch, time, features]
    # ])
    # lstm_model = NormedLSTM(16)
    lstm_model = FeedBack(32, 15, 32)

    history = compile_and_fit(lstm_model, wnd, 200, 10)

    val_performance = lstm_model.evaluate(wnd.validate_ds)
    performance = lstm_model.evaluate(wnd.test_ds, verbose=0)

    wnd.plot(lstm_model, plot_col_index=0)
    # wnd.plot(lstm_model, 'Close')

    # df = fracDiff_FFD(vwap, 0.19)
    df = fracDiff_FFD(np.log(vwap[2:5]), 1.)

    sig = df[:390]
    t, dt = np.linspace(0, sig.shape[0]*6.5/390, sig.shape[0], retstep=True)
    fs = 1/dt
    w = 6.5/390
    freq = np.linspace(1, 2*fs/25, 100)
    widths = w*fs / (2*freq*np.pi)
    cwtm = signal.cwt(sig, signal.ricker, widths)#, w=w)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, sig)
    axs[0].set_xlim(min(t), max(t))
    axs[1].pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
    axs[1].set_ylim(1, 3)
    fig.show()

    exit(0)

