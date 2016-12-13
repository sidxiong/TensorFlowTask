import cPickle as pkl
import datetime
import time
import os

import numpy as np
from model import TextClassifier
import tensorflow as tf

def train(x_train, y_train, x_valid, y_valid, num_epoch=50, batch_size=128, dropout=0.5,
          filter_size=[3, 4, 5], embedding_dim=128, num_filters=128, dirname=None):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = TextClassifier(
                input_size=x_train.shape[1],
                output_size=y_train.shape[1],
                vocab_size=20000,
                embedding_size=embedding_dim,
                filter_widths=filter_size,
                num_filters=num_filters,
                dropout=dropout)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)

            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                         tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if dirname is None:
                dirname = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)


            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "runs/1481443544/checkpoints/model-600")
            #print("Model restored.")

            def get_batch(x_data, y_data, batch_size=128):
                size = len(x_train)
                for k in xrange(size / batch_size):
                    yield x_data[k*batch_size: (k+1)*batch_size],\
                          y_data[k*batch_size: (k+1)*batch_size]


            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout: dropout
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def valid_step():
                """
                Evaluates model on a validation set in batch
                """
                batch_iter = get_batch(x_valid, y_valid, batch_size)
                losses = []
                accuracies = []

                for x_batch, y_batch in batch_iter:
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout: 1
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    losses.append(loss)
                    accuracies.append(accuracy)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, np.mean(losses), np.mean(accuracies)))
                dev_summary_writer.add_summary(summaries, step)



            for epoch in xrange(num_epoch):
                batch_iter = get_batch(x_train, y_train, batch_size)
                for x_batch, y_batch in batch_iter:
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % 50 == 0:
                        print("\nEvaluation:")
                        dev_step(x_valid, y_valid, writer=dev_summary_writer)
                        print("")
                    if current_step % 100 == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

def test(test_X, test_y, restored_model_file, filter_size=[3, 4, 5],
          embedding_dim=128, num_filters=128):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextClassifier(
                input_size=test_X.shape[1],
                output_size=test_y.shape[1],
                vocab_size=20000,
                embedding_size=embedding_dim,
                filter_widths=filter_size,
                num_filters=num_filters,
                dropout=0)

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, restored_model_file)

            def get_batch(x_data, y_data, batch_size=128):
                size = len(x_data)
                for k in xrange(size / batch_size):
                    yield x_data[k*batch_size: (k+1)*batch_size],\
                          y_data[k*batch_size: (k+1)*batch_size]

            batch_iter = get_batch(test_X, test_y, 128)
            losses = []
            accuracies = []

            for x_batch, y_batch in batch_iter:
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout: 0
                }
                loss, accuracy = sess.run([cnn.loss, cnn.accuracy], feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)

                loss, accuracy = sess.run([cnn.loss, cnn.accuracy], feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)

            time_str = datetime.datetime.now().isoformat()
            print("{}: loss {:g}, acc {:g} (on test)".format(time_str, np.mean(losses), np.mean(accuracies)))

if __name__ == "__main__":
    with open('data_split.p', 'rb') as f:
        x_train, y_train, x_valid, y_valid, x_test, y_test = pkl.load(f)
    train(x_train, y_train, x_valid, y_valid)
    #test(x_test, y_test, 'runs/1481424646/checkpoints/model-4900')
