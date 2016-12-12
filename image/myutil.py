import os
import numpy as np
import tensorflow as tf
from scipy.io import *
from scipy.misc import *

def create_array(n_sample, prefix, size=(128, 128, 3)):
    array = np.empty((n_sample, size[0], size[1], size[2]))
    _l = os.listdir(prefix)
    if len(_l) > n_sample:
        _l = _l[1:]
        assert len(_l) == n_sample

    for i, f in enumerate(_l):
        if i % 100 == 0:
            print i,
        _ = imresize(imread(prefix + f), (size[0], size[1])) / 255.
        if _.shape != size:
            _ = np.stack((_, _, _), axis=2)
        _mean = np.mean(_)
        array[i] = _ - _mean
    print "--"
    return array

def create_train_array(n_sample, prefix, size=(128,128,3)):
    array = np.empty((n_sample*6, size[0], size[1], size[2]))
    _l = os.listdir(prefix)
    if len(_l) > n_sample:
        _l = _l[1:]
        assert len(_l) == n_sample
    for i, f in enumerate(_l):
        if i % 100 == 0:
            print i, 
        _ = imresize(imread(prefix + f), (size[0], size[1])) / 255.
        if _.shape != size:
            _ = np.stack((_, _, _), axis=2)
        a = np.fliplr(_)
        b = np.flipud(_)
        c = imrotate(_, 90.) / 255.
        d = imrotate(_, -90.) / 255.
        e = imrotate(_, 180.) / 255.

        _mean = np.mean(_)
        array[i * 6] = _ - _mean
        array[i * 6 + 1] = a - _mean
        array[i * 6 + 2] = b - _mean
        array[i * 6 + 3] = c - _mean
        array[i * 6 + 4] = d - _mean
        array[i * 6 + 5] = e - _mean

    print "--"
    return array

def build_default_dataset( size=(227,227,3) ):
    n_train, n_valid, n_test = 800, 300, 200
    bird_train = create_train_array(n_train, 'bird/train/', size=size)
    bird_valid = create_array(n_valid, 'bird/validation/', size=size)
    bird_test = create_array(n_test, 'bird/test/', size=size)
    fish_train = create_train_array(n_train, 'fish/train/', size=size)
    fish_valid = create_array(n_valid, 'fish/validation/', size=size)
    fish_test = create_array(n_test, 'fish/test/', size=size)

    _pos, _neg = np.ones((n_train*6,)), np.zeros((n_train*6,))
    y_bird_train = np.column_stack((_pos, _neg))
    y_fish_train = np.column_stack((_neg, _pos))

    _pos, _neg = np.ones((n_valid,)), np.zeros((n_valid,))
    y_bird_valid = np.column_stack((_pos, _neg))
    y_fish_valid = np.column_stack((_neg, _pos))

    _pos, _neg = np.ones((n_test,)), np.zeros((n_test,))
    y_bird_test = np.column_stack((_pos, _neg))
    y_fish_test = np.column_stack((_neg, _pos))

    idx = np.arange(n_train * 6  * 2); np.random.shuffle(idx)
    train_X = np.concatenate((bird_train, fish_train))[idx]
    train_y = np.concatenate((y_bird_train, y_fish_train))[idx]

    idx = np.arange(n_valid * 2); np.random.shuffle(idx)
    valid_X = np.concatenate((bird_valid, fish_valid))[idx]
    valid_y = np.concatenate((y_bird_valid, y_fish_valid))[idx]

    idx = np.arange(n_test * 2); np.random.shuffle(idx)
    test_X = np.concatenate((bird_test, fish_test))[idx]
    test_y = np.concatenate((y_bird_test, y_fish_test))[idx]

    print train_X.shape, train_y.shape
    print valid_X.shape, valid_y.shape
    print test_X.shape, test_y.shape

    return train_X, train_y,\
           valid_X, valid_y,\
           test_X, test_y

def batch_generator(train_X, train_y, batch_size=128):
    size = train_X.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    _X = train_X[idx]
    _y = train_y[idx]
    for k in xrange(size / batch_size):
        yield _X[k*batch_size : (k+1) * batch_size], \
              _y[k*batch_size : (k+1) * batch_size]


def conv(input, kernel, bias, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, bias), [-1]+conv.get_shape().as_list()[1:])

