import os
import logging
import numpy as np

from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import fasttext

from .utils import file_ops
from .utils.config import DATA_PATH


def gen_data():
    x_trn, y_trn, categories = file_ops.read_file(
        os.path.join(DATA_PATH, "ner_train.txt")
    )
    x_trn, x_val, y_trn, y_val = train_test_split(
        x_trn, y_trn, test_size=0.20, random_state=42
    )
    x_test, y_test, _ = file_ops.read_file(os.path.join(DATA_PATH, "ner_test.txt"))

    return {
        "train": (x_trn, y_trn, categories),
        "valid": (x_val, y_val),
        "test": (x_test, y_test),
    }
