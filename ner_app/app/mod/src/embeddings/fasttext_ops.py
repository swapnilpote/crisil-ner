import fasttext
import numpy as np

from tensorflow import keras


def load_fasttext(fpath: str) -> object:
    fasttext_model = fasttext.load_model(fpath)

    return fasttext_model


def fasttext_generator(
    fasttext_obj: object, cat2id: dict, x: list, y: list, maxlen: int, categories: list
) -> tuple:
    x_vector, y_vector = [], []

    x = keras.preprocessing.sequence.pad_sequences(
        sequences=x,
        maxlen=maxlen,
        padding="post",
        truncating="post",
        value="pad",
        dtype=object,
    )
    y = keras.preprocessing.sequence.pad_sequences(
        sequences=y,
        maxlen=maxlen,
        padding="post",
        truncating="post",
        value="pad",
        dtype=object,
    )

    for X, Y in zip(x, y):
        tmp_x = []
        tmp_y = []
        for word, label in zip(X, Y):
            tmp_x.append(fasttext_obj.get_word_vector(word))
            tmp_y.append(keras.utils.to_categorical(cat2id.get(label), len(categories)))

        x_vector.append(tmp_x)
        y_vector.append(tmp_y)

    return np.asarray(x_vector), np.asarray(y_vector)