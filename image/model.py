#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'sidxiong'

import tensorflow as tf
import numpy as np
from myutil import conv

class AlexNet(object):
    def __init__(self, size=(227, 227, 3)):
        self.X = tf.placeholder(tf.float32, shape=(None,) + size)
        self.y = tf.placeholder(tf.float32, shape=(None, 2))

        pretrained_AlexNet = np.load('bvlc_alexnet.npy').item()
        conv1_W = pretrained_AlexNet['conv1'][0]
        conv1_b = pretrained_AlexNet['conv1'][1]
        conv2_W = pretrained_AlexNet['conv2'][0]
        conv2_b = pretrained_AlexNet['conv2'][1]
        conv3_W = pretrained_AlexNet['conv3'][0]
        conv3_b = pretrained_AlexNet['conv3'][1]
        conv4_W = pretrained_AlexNet['conv4'][0]
        conv4_b = pretrained_AlexNet['conv4'][1]
        conv5_W = pretrained_AlexNet['conv5'][0]
        conv5_b = pretrained_AlexNet['conv5'][1]
        
        # conv1 + relu1 + lrn1 + pool1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(conv1_W, name='conv1_W', trainable=False)
            _bias = tf.Variable(conv1_b, name='conv1_b', trainable=False)
            _conv1 = conv(self.X, kernel, _bias, 11, 11, 96, 4, 4, padding='VALID', group=1)
            conv1 = tf.nn.relu(_conv1, name=scope)

        r = 2; alpha = 2e-5; beta = 0.75; bias = 1.
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=r,
                                                         alpha=alpha,
                                                         beta=beta,
                                                         bias=bias, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], 
                                         strides=[1,2,2,1], 
                                         padding='VALID', name='pool1')

        # conv2 + relu2 + pool2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(conv2_W, name='conv2_W', trainable=False)
            _bias = tf.Variable(conv2_b, name='conv2_b', trainable=False)
            _conv2 = conv(pool1, kernel, _bias, 5, 5, 256, 1, 1, padding='SAME', group=2)
            conv2 = tf.nn.relu(_conv2, name=scope)
            
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],
                                      strides=[1,2,2,1],
                                      padding='VALID',
                                      name='pool2')

        # conv3 + conv4 + conv5
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(conv3_W, name='conv3_W', trainable=False)
            _bias = tf.Variable(conv3_b, name='conv3_b', trainable=False)
            _conv3 = conv(pool2, kernel, _bias, 3, 3, 384, 1, 1, padding='SAME')
            conv3 = tf.nn.relu(_conv3, name=scope)

        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(conv4_W, name='conv4_W', trainable=False)
            _bias = tf.Variable(conv4_b, name='conv4_b', trainable=False)
            _conv4 = conv(conv3, kernel, _bias, 3, 3, 384, 1, 1, padding='SAME', group=2)
            conv4 = tf.nn.relu(_conv4, name=scope)

        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(conv5_W, name='conv5_W', trainable=False)
            _bias = tf.Variable(conv5_b, name='conv5_b', trainable=False)
            _conv5 = conv(conv4, kernel, _bias, 3, 3, 256, 1, 1, padding='SAME', group=2)
            conv5 = tf.nn.relu(_conv5, name=scope)
        self.pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
        
        _ = map(lambda x:x.value, self.pool5.get_shape()[1:])
        self.n_pool5 = np.prod(_)
        self.l2_loss_reg = tf.constant(0.)

class AN_3FC(AlexNet):
    def __init__(self):
        super(AN_3FC, self).__init__()
        self.keep_prob_fc6 = tf.placeholder(tf.float32)
        self.keep_prob_fc7 = tf.placeholder(tf.float32)
        self.keep_prob_fc8 = tf.placeholder(tf.float32)

        n_fc6 = 2048
        n_fc7 = 1024
        n_fc8 = 256
        n_fc9 = 2

        with tf.name_scope('fc6') as scope:
            _W = tf.Variable(tf.truncated_normal((self.n_pool5, n_fc6)) / np.sqrt(n_fc6), name='fc6_W')
            _b = tf.Variable(tf.truncated_normal((n_fc6,)) / np.sqrt(n_fc6), name='fc6_b')
            fc6 = tf.nn.relu_layer(tf.reshape(self.pool5, [-1, self.n_pool5]), _W, _b)
            fc6_drop = tf.nn.dropout(fc6, self.keep_prob_fc6)
          
            self.l2_loss_reg += tf.nn.l2_loss(_W)
            self.l2_loss_reg += tf.nn.l2_loss(_b)
        with tf.name_scope('fc7') as scope:
            _W = tf.Variable(tf.truncated_normal((n_fc6, n_fc7)) / np.sqrt(n_fc7), name='fc7_W')
            _b = tf.Variable(tf.truncated_normal((n_fc7,)) / np.sqrt(n_fc7), name='fc7_b')
            fc7 = tf.nn.relu_layer(fc6_drop, _W, _b)
            fc7_drop = tf.nn.dropout(fc7, self.keep_prob_fc7)
          
            self.l2_loss_reg += tf.nn.l2_loss(_W)
            self.l2_loss_reg += tf.nn.l2_loss(_b)
        with tf.name_scope('fc8') as scope:
            _W = tf.Variable(tf.truncated_normal((n_fc7, n_fc8)) / np.sqrt(n_fc8), name='fc8_W')
            _b = tf.Variable(tf.truncated_normal((n_fc8,)) / np.sqrt(n_fc8), name='fc8_b')
            fc8 = tf.nn.relu_layer(fc7_drop, _W, _b)
            fc8_drop = tf.nn.dropout(fc8, self.keep_prob_fc8)
          
            self.l2_loss_reg += tf.nn.l2_loss(_W)
            self.l2_loss_reg += tf.nn.l2_loss(_b)

        with tf.name_scope('out') as scope:
            _W = tf.Variable(tf.truncated_normal((n_fc8, n_fc9)) / np.sqrt(n_fc9), name='fc9_W')
            _b = tf.Variable(tf.truncated_normal((n_fc9,)) / np.sqrt(n_fc9), name='fc9_b')
            fc9 = tf.nn.xw_plus_b(fc8_drop, _W, _b)
          
            self.score = fc9
            self.prediction = tf.argmax(fc9, 1)
          
            self.l2_loss_reg += tf.nn.l2_loss(_W)
            self.l2_loss_reg += tf.nn.l2_loss(_b)

        with tf.name_scope('loss') as scope:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc9, self.y))
            self.loss = cross_entropy + 0.005 * self.l2_loss_reg
            
        with tf.name_scope('accuracy') as scope:
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')



class AN_Direct(AlexNet):
    def __init__(self):
        super(AN_Direct, self).__init__()
        self.keep_prob_fc6 = tf.placeholder(tf.float32)

        n_fc7 = 2
        with tf.name_scope('out') as scope:
            pool5_drop = tf.nn.dropout(tf.reshape(self.pool5, [-1, self.n_pool5]), self.keep_prob_fc6)
            _W = tf.Variable(tf.truncated_normal((self.n_pool5, n_fc7)) / np.sqrt(n_fc7), name='fc7_W')
            _b = tf.Variable(tf.truncated_normal((n_fc7,)) / np.sqrt(n_fc7), name='fc7_b')
            fc7 = tf.nn.xw_plus_b(pool5_drop, _W, _b)
            
            self.score = fc7
            self.prediction = tf.argmax(fc7, 1)
            
            self.l2_loss_reg += tf.nn.l2_loss(_W)
            self.l2_loss_reg += tf.nn.l2_loss(_b)
            
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc7, self.y))
            self.loss = cross_entropy + 0.02 * self.l2_loss_reg
            
        with tf.name_scope('accuracy') as scope:
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


# following are simple CNN definitions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

class SimpleCNN(object):
    def __init__(self, size=(128, 128, 3)):
        self.X = tf.placeholder(tf.float32, shape=(None,) + size)
        self.y = tf.placeholder(tf.float32, shape=(None, 2))
        self.keep_prob_1 = tf.placeholder(tf.float32)
        self.keep_prob_2 = tf.placeholder(tf.float32)

        self.l2_loss_reg = tf.constant(0.)

        with tf.name_scope('conv1') as scope:
            W_conv1 = weight_variable([5, 5, 3, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(self.X, W_conv1) + b_conv1)
            self.l2_loss_reg += tf.nn.l2_loss(W_conv1)
            self.l2_loss_reg += tf.nn.l2_loss(b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('conv2') as scope:
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            self.l2_loss_reg += tf.nn.l2_loss(W_conv2)
            self.l2_loss_reg += tf.nn.l2_loss(b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fc1') as scope:
            W_fc1 = weight_variable([32 * 32 * 64, 256])
            b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 32 * 32 * 64]), W_fc1) + b_fc1)
            self.l2_loss_reg += tf.nn.l2_loss(W_fc1)
            self.l2_loss_reg += tf.nn.l2_loss(b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob_1)

        with tf.name_scope('fc2') as scope:
            W_fc2 = weight_variable([256, 64])
            b_fc2 = bias_variable([64])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            self.l2_loss_reg += tf.nn.l2_loss(W_fc2)
            self.l2_loss_reg += tf.nn.l2_loss(b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob_2)

        with tf.name_scope('fc3') as scope:
            W_fc3 = weight_variable([64, 2])
            b_fc3 = bias_variable([2])
            y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
            self.l2_loss_reg += tf.nn.l2_loss(W_fc3)
            self.l2_loss_reg += tf.nn.l2_loss(b_fc3)

        self.score = y_conv
        self.prediction = tf.argmax(y_conv, 1)

        with tf.name_scope('loss') as scope:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, self.y))
            self.loss = cross_entropy + 0.005 * self.l2_loss_reg
            
        with tf.name_scope('accuracy') as scope:
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


if __name__ == "__main__":
    a = AN_3FC()
    b = AN_Direct()
    c = SimpleCNN()
