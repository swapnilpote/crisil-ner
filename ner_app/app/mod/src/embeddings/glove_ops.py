import numpy as np

from tensorflow import keras


def load_glove(fpath: str, dim: int) -> tuple:
    with open(fpath, "r") as f:
        lines = f.readlines()

    matrix = np.zeros((len(lines), dim))
    id2word = dict()
    word2id = dict()

    for idx, line in enumerate(lines):
        line_split = line.split()
        word = line_split[0]

        matrix[idx] = line_split[1:]

        id2word[idx] = word
        word2id[word] = idx

    return matrix, id2word, word2id


def glove_generator(
    word2id: dict,
    cat2id: dict,
    x: list,
    y: list,
    maxlen: int,
    categories: list,
    data_type="test",
) -> tuple:
    x_vector, y_vector = [], []

    if data_type != "train":
        y = [["o"] * len(x[0])]

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
            tmp_x.append(word2id.get(word, word2id.get("unk")))
            tmp_y.append(keras.utils.to_categorical(cat2id.get(label), len(categories)))

        x_vector.append(tmp_x)
        y_vector.append(tmp_y)

    return np.asarray(x_vector), np.asarray(y_vector)