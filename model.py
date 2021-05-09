from numpy import reshape
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dropout  # @TODO add Activation and TimeDistributed layers
from keras.utils import plot_model


class ModelArtifact:

    def __init__(self):
        self.model = Sequential()

    def build(self, x_data, y_data, n_patterns, seq_length):
        X = reshape(x_data, (n_patterns, seq_length, 1))
        y = to_categorical(y_data)

        self.model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(y.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

        checkpoint = ModelCheckpoint('model_weights_saved.hdf5', monitor='loss', verbose=1, save_best_only=True,
                                     mode='min')
        self.model.fit(X, y, epochs=250, batch_size=256, callbacks=[checkpoint])
        plot_model(self.model, to_file='model_plot_dec.png', show_shapes=True, show_layer_names=True)

        return self.model.summary()

    def get(self):
        return self.model.load_weights('model_weights_saved.hdf5')

    def __iter__(self):
        # @TODO implement it
        raise NotImplementedError
