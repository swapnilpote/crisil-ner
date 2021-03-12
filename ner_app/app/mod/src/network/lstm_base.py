import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .metrics import get_f1


def lstm_glove_model(matrix: np.ndarray, categories: list, maxlen: int) -> object:

    input_layer = keras.Input(shape=(maxlen,))
    embedding_layer = keras.layers.Embedding(
        400000, 100, weights=[matrix], input_length=maxlen, trainable=False
    )(input_layer)
    bi_lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(maxlen, return_sequences=True, recurrent_dropout=0.5)
    )(embedding_layer)
    dense_layer = keras.layers.Dense(len(categories), activation="softmax")(
        bi_lstm_layer
    )

    model = keras.Model(input_layer, dense_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.02, reduction=tf.keras.losses.Reduction.SUM
        ),
        metrics=[get_f1, "accuracy"],
    )

    return model


def lstm_fasttext_model(categories: list, maxlen: int) -> object:
    input_layer = keras.Input(shape=(maxlen, 100))
    bi_lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(maxlen, return_sequences=True, recurrent_dropout=0.5)
    )(input_layer)
    dense_layer = keras.layers.Dense(len(categories), activation="softmax")(
        bi_lstm_layer
    )

    model = keras.Model(input_layer, dense_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=0.02, reduction=tf.keras.losses.Reduction.SUM
        ),
        metrics=[get_f1, "accuracy"],
    )
    return model
