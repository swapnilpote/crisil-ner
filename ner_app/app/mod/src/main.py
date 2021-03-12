import os

from sklearn.model_selection import train_test_split

from tensorflow import keras

from .utils.config import DATA_PATH
from .utils import file_ops

from .embeddings import glove_ops, fasttext_ops

from .network import lstm_base

glove_fpath, glove_fdownloaded, glove_funzipped = file_ops.get_file(
    os.path.join(DATA_PATH, "glove.6B.zip"), "http://nlp.stanford.edu/data/glove.6B.zip"
)
fasttext_fpath, fasttext_fdownloaded, fasttext_funzipped = file_ops.get_file(
    os.path.join(DATA_PATH, "cc.en.100.bin"),
    "https://geniekalp.s3.ap-south-1.amazonaws.com/pretrained_embeddings/cc.en.100.bin",
)

glove_matrix, glove_id2word, glove_word2id = glove_ops.load_glove(
    os.path.join(DATA_PATH, "glove.6B.100d.txt"), 100
)

fasttext_model = fasttext_ops.load_fasttext(os.path.join(DATA_PATH, fasttext_fpath))

x_trn, y_trn, categories = file_ops.read_file(os.path.join(DATA_PATH, "ner_train.txt"))

x_trn, x_val, y_trn, y_val = train_test_split(
    x_trn, y_trn, test_size=0.20, random_state=42
)

x_test, y_test, _ = file_ops.read_file(os.path.join(DATA_PATH, "ner_test.txt"))

cat2id = {k: i for i, k in enumerate(categories)}
id2cat = {i: k for i, k in enumerate(categories)}

maxlen = 47

x_trn_glove, y_trn_glove = glove_ops.glove_generator(
    glove_word2id, cat2id, x_trn, y_trn, maxlen, categories
)
x_val_glove, y_val_glove = glove_ops.glove_generator(
    glove_word2id, cat2id, x_val, y_val, maxlen, categories
)
x_test_glove, y_test_glove = glove_ops.glove_generator(
    glove_word2id, cat2id, x_test, y_test, maxlen, categories
)

glove_lstm_base_model = lstm_base.lstm_glove_model(glove_matrix, categories, maxlen)

checkpoints = keras.callbacks.ModelCheckpoint(
    "best_glove_model.hdf5",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

batch_size = 64
epochs = 10
verbose = 1
training_records = x_trn_glove.shape[0]

glove_lstm_base_model_history = glove_lstm_base_model.fit(
    x_trn_glove,
    y_trn_glove,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    callbacks=[checkpoints],
    validation_data=(x_val_glove, y_val_glove),
    steps_per_epoch=training_records // batch_size,
)
