import tensorflow as tf

class TextClassifier(object):

    def __init__(self, input_size, output_size, vocab_size, embedding_size, filter_widths, num_filters, dropout):
        # Dummy vars to store input and output
        self.input_x = tf.placeholder(tf.int32, [None, input_size], name='input_x') # text sequence input
        self.input_y = tf.placeholder(tf.float32, [None, output_size], name='input_y') # classes output
        self.dropout = self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

        l2_loss = tf.constant(0.0)

        # Embedding layer
        # Force cpu here because embedding layer has no gpu support
        with tf.device('/cpu:0'), tf.name_scope('embedding-layer'):
            # random initialization of weights
            # change this line to load pretrained weights
            embedding_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                            name='embedding-W')
            self.embedding_layer = tf.nn.embedding_lookup(embedding_W, self.input_x)
            # reshape to word with conv2d layer
            self.embedding_reshaped = tf.expand_dims(self.embedding_layer, -1)

        pool_layer_out = []
        input_channel = 1
        for i, filter_width in enumerate(filter_widths):
            with tf.name_scope('conv-pool-layer%s' % i):
                # Convolution layer
                filter_shape = [filter_width, embedding_size, input_channel, num_filters]
                # random initialization of weights
                conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.2), name='W_conv%s' % i)
                conv_b = tf.Variable(tf.constant(0.2, shape=[num_filters]), name='b_conv%s' % i)
                # stride of 1 is normally used in text
                # dont do convolutions outside of boundary
                conv = tf.nn.conv2d(
                    self.embedding_reshaped, conv_W, strides=[1, 1, 1, 1], padding='VALID', name='conv%s' % i)
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu%s' % i)

                # Pooling layer
                pooling = tf.nn.max_pool(
                    conv_relu,
                    ksize=[1, input_size - filter_width + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool%s' % i)
                pool_layer_out.append(pooling)

        num_filters_total = num_filters * len(filter_widths)
        self.pool_layers = tf.concat(3, pool_layer_out)
        self.pool_layers_flat = tf.reshape(self.pool_layers, [-1, num_filters_total])

        # Dropout for regularization
        with tf.name_scope('dropout-layer'):
            self.dropout_layer = tf.nn.dropout(self.pool_layers_flat, 1 - self.dropout, name='dropout')

        # Final output layer
        with tf.name_scope("output-layer"):
            # use shared variable here
            output_W = tf.get_variable("final-W",
                initializer=tf.truncated_normal([num_filters_total, output_size], stddev=0.2))
            output_b = tf.Variable(tf.constant(0.2, shape=[output_size]), name="final-b")
            l2_loss += tf.nn.l2_loss(output_W)
            l2_loss += tf.nn.l2_loss(output_b)
            self.scores = tf.nn.xw_plus_b(self.dropout_layer, output_W, output_b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # Objective to minimize
        with tf.name_scope("loss"):
            # mean cross-entropy loss
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + 0.1 * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")