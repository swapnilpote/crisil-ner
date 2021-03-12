import os
import numpy as np

from .utils.config import DATA_PATH
from .utils import file_ops

from .embeddings import glove_ops
from .network import lstm_base


class Prediction:
    def __init__(self) -> None:
        glove_fpath, glove_fdownloaded, glove_funzipped = file_ops.get_file(
            os.path.join(DATA_PATH, "glove.6B.zip"),
            "http://nlp.stanford.edu/data/glove.6B.zip",
        )

        (
            self.glove_matrix,
            self.glove_id2word,
            self.glove_word2id,
        ) = glove_ops.load_glove(os.path.join(DATA_PATH, "glove.6B.100d.txt"), 100)

        _, _, self.categories = file_ops.read_file(
            os.path.join(DATA_PATH, "ner_train.txt")
        )

        self.cat2id = {k: i for i, k in enumerate(self.categories)}
        self.id2cat = {i: k for i, k in enumerate(self.categories)}

        self.maxlen = 47

        self.glove_lstm_model = lstm_base.lstm_glove_model(
            self.glove_matrix, self.categories, self.maxlen
        )
        self.glove_lstm_model.load_weights(
            os.path.join(DATA_PATH, "best_glove_model.hdf5")
        )

    def glove_predict(self, x: str) -> dict:
        x_glove, _ = glove_ops.glove_generator(
            self.glove_word2id,
            self.cat2id,
            [[word for word in x.split()]],
            None,
            self.maxlen,
            self.categories,
        )

        result = self.glove_lstm_model.predict(x_glove).argmax(-1)[0]
        result = [self.id2cat[i] for i in result]

        return result