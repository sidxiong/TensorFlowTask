from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np

import cPickle as pkl
import os

def extract_data(dir='20news-18828', top_k_words=20000, max_sequence_length=1000):
    label_ids = {}

    label_idx = 0
    xs = []
    ys = []
    for folder in os.listdir(dir):
        news_folder = os.path.join(dir, folder)
        if os.path.isdir(news_folder):
            label_ids[folder] = label_idx
            for news_title in os.listdir(news_folder):
                if news_title.isdigit():
                    news_item_path = os.path.join(news_folder, news_title)
                    with open(news_item_path) as f:
                        xs.append(f.read())
                        ys.append(label_idx)
            label_idx += 1

    tokenizer = Tokenizer(nb_words=top_k_words)
    tokenizer.fit_on_texts(xs)
    text_sequences = tokenizer.texts_to_sequences(xs)

    print "Found %s unique words." % len(tokenizer.word_index)
    print "Median length of docs: %s" % np.median((map(len, text_sequences)))

    xs_padded = pad_sequences(text_sequences, maxlen=max_sequence_length)
    ys_categorical = to_categorical(np.asarray(ys))

    indices = list(xrange(len(ys_categorical)))
    np.random.shuffle(indices)
    train, validate, test = np.split(indices, [int(.6*len(ys_categorical)), int(.8*len(ys_categorical))])
    x_train, y_train = xs_padded[train], ys_categorical[train]
    x_valid, y_valid = xs_padded[validate], ys_categorical[validate]
    x_test, y_test = xs_padded[test], ys_categorical[test]

    with open('data_split.p', 'wb') as out:
        pkl.dump((x_train, y_train, x_valid, y_valid, x_test, y_test), out)

if __name__ == '__main__':
    extract_data()
