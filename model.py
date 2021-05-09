from numpy import reshape
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout  # @TODO add Activation and TimeDistributed layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from tensorflow.python.keras.optimizer_v2.adam import Adam


def build_model(x_train_pad, y_train, max_tokens):
    model = Sequential()
    model.add(Embedding(input_dim=10000,
                        output_dim=50,
                        input_length=max_tokens,
                        name='embedding_layer'))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(LSTM(units=8, return_sequences=True))
    model.add(LSTM(units=4))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train_pad, y_train, epochs=250, batch_size=256)
    model.save("bot_model.h5")
    return model, history


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
