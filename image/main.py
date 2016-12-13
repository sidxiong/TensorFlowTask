#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'sidxiong'

import numpy as np
import tensorflow as tf
from myutil import *
from model import *
import time, datetime
import sys, os

def train(train_X, train_y, valid_X, valid_y, net_name='an_direct', restored_model_file=None, learning_rate=1e-4, dirname=None, n_epoch=50, batch_size=128):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if net_name == 'an_fc3':
                net = AN_3FC()
            elif net_name == 'an_direct':
                net = AN_Direct()
            else:
                net = SimpleCNN()

            global_step = tf.Variable(0, name='global_step', trainable=False)
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
            if dirname is None:
                dirname = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)
            
            loss_summary = tf.scalar_summary("loss", net.loss)
            acc_summary = tf.scalar_summary("accuracy", net.accuracy)
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
            
            saver = tf.train.Saver(tf.all_variables())
            
            if restored_model_file is None:
                sess.run(tf.initialize_all_variables())
            else:
                saver.restore(sess, restored_model_file)
                print 'model restored!'
            

            if net_name == 'an_fc3':
                _fd_train = {net.keep_prob_fc6: 0.5, net.keep_prob_fc7: 0.5, net.keep_prob_fc8: 0.5}
                _fd_valid = {net.keep_prob_fc6: 1., net.keep_prob_fc7: 1., net.keep_prob_fc8: 1.}
            elif net_name == 'an_direct':
                _fd_train = {net.keep_prob_fc6: 0.5}
                _fd_valid = {net.keep_prob_fc6: 1.}
            else:
                _fd_train = {net.keep_prob_1: 0.5, net.keep_prob_2: 0.5}
                _fd_valid = {net.keep_prob_1: 1., net.keep_prob_2: 1.}
                
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  net.X: x_batch,
                  net.y: y_batch
                }
                feed_dict.update(_fd_train)
                
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, net.loss, net.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                
                if step % 10 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    
                train_summary_writer.add_summary(summaries, step)
                
            def valid_step(writer=None):
                feed_dict = {
                  net.X: valid_X,
                  net.y: valid_y
                }
                feed_dict.update(_fd_valid)

                step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, net.loss, net.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g} (on validation)".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                
            
            # do training!
            for epoch in range(n_epoch):
                batch_iter = batch_generator(train_X, train_y, batch_size)
                for i, (_X, _y) in enumerate(batch_iter):
                    train_step(_X, _y)
                print 'complete epoch is: %d' % (epoch + 1)
                
                valid_step(writer=dev_summary_writer)
                if (epoch + 1) % 10 == 0:
                    curr_step = tf.train.global_step(sess, global_step)
                    ts = str(int(time.time()))
                    saver.save(sess, checkpoint_prefix, global_step=curr_step)
                    print 'model checkpoint saved at step %d' % curr_step

def test(test_X, test_y, restored_model_file, net_name='an_direct'):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            if net_name == 'an_fc3':
                net = AN_3FC()
                _fd_test = {net.keep_prob_fc6: 1., net.keep_prob_fc7: 1., net.keep_prob_fc8: 1.}
            elif net_name == 'an_direct':
                net = AN_Direct()
                _fd_test = {net.keep_prob_fc6: 1.}
            else:
                net = SimpleCNN()
                _fd_test = {net.keep_prob_1: 1., net.keep_prob_2: 1.}
            
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, restored_model_file)

            feed_dict = {
              net.X: test_X,
              net.y: test_y
            }
            # update feed dict to adapt different networks
            feed_dict.update(_fd_test)
            loss, accuracy = sess.run([net.loss, net.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g} (on test)".format(time_str, loss, accuracy))




if __name__ == "__main__":
    train_X, train_y, valid_X, valid_y, test_X, test_y = \
           build_default_dataset((128, 128, 3))

    train(train_X, train_y, valid_X, valid_y,
          net_name='simple', learning_rate=1e-3, dirname='simple-cnn', n_epoch=30, batch_size=128)
