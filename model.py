import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
from sample_input import sample_input
from scaler import Scaler

import warnings

warnings.filterwarnings('ignore')

KERNEL_WIDTH = 1
LABEL_WIDTH = 144
INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1
LABEL_COLUMNS = ['capacity']


class DataWindow:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, scaler,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.scaler = scaler
        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift


            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

            def split_to_inputs_labels(self, features):
                inputs = features[:, self.input_slice, :]
                labels = features[:, self.labels_slice, :]
                if self.label_columns is not None:
                    labels = tf.stack(
                        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                        axis=-1
                    )
                    inputs.set_shape([None, self.input_width, None])
                    labels.set_shape([None, self.label_width, None])

                return inputs, labels

            def make_dataset(self, data):
                data = np.array(data, dtype=np.float32)
                ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                    data=data,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=1,
                    shuffle=True,
                    batch_size=32

                )
                ds = ds.map(self.split_to_inputs_labels)
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
            def sample_batch(self):
                result = getattr(self, '_sample_batch', None)
                if result is None:
                    result = next(iter(self.train))
                    self._sample_batch = result
                return result

        DataWindow.train = train
        DataWindow.val = val
        DataWindow.test = test
        DataWindow.sample_batch = sample_batch
        DataWindow.make_dataset = make_dataset
        DataWindow.split_to_inputs_labels = split_to_inputs_labels


def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


def generate_model(train_df,
                   test_df,
                   val_df,
                   scaler,
                   input_width=INPUT_WIDTH,
                   label_width=LABEL_WIDTH,
                   shift=LABEL_WIDTH,
                   label_columns=LABEL_COLUMNS,
                   kernel_width=KERNEL_WIDTH,
                   file_name='../model/model.h5'):
    # TODO: Read label_columns from arg.
    train_df = scaler.transform_df(train_df)
    test_df = scaler.transform_df(test_df)
    val_df = scaler.transform_df(val_df)

    cnn_multi_window = DataWindow(input_width, label_width, shift,
                                  train_df, test_df, val_df, scaler,
                                   label_columns)
    cnn_model = Sequential([
        Conv1D(32, activation='relu', kernel_size=kernel_width),
        Dense(units=32, activation='relu'),
        Dense(1, kernel_initializer=tf.initializers.zeros)
    ])
    compile_and_fit(cnn_model, cnn_multi_window)
    cnn_model.save(file_name)


if __name__ == '__main__':
    result = sample_input(sys.argv[1])
    scaler = Scaler(result[0])
    generate_model(result[0], result[1], result[2], scaler=scaler, file_name='../model/model' + result[3] + ".h5")
