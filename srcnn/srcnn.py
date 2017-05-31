"""
the model of SRCNN
PAPER : Image Super-Resolution Using Deep Convolutional Networks
"""

import os
import time
import sys

import numpy as np
import tensorflow as tf
import utils


def _maybe_pad_x(x, padding, is_training):
    if padding == 0:
        x_pad = x
    elif padding>0:
        x_pad = tf.cond(is_training, lambda : x, lambda : utils.replicate_padding(x, padding))
    else:
        raise ValueError('Padding value %i should be greater than or equal to 1' % padding)
    return x_pad


class SRCNN:

    def __init__(self, layer_sizes, filter_sizes, input_depth=1,
                 learning_rate=1e-4,
                 gpu=True,
                 upscale_factor=2):
        self.upscale_factor = upscale_factor
        self.layer_sizes = layer_sizes
        self.filter_sizes = filter_sizes
        self.input_depth = input_depth
        self.learning_rate = learning_rate
        if gpu:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

        self.global_step = tf.Variable(0, trainable=False)
        self._build_graph()
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                        100000, 0.96)

    def _set_placeholders(self):
        with tf.name_scope('placeholders'):
            self.images = tf.placeholder(tf.float32, shape=(None, None, None, self.input_depth), name='input')
            self.labels = tf.placeholder(tf.float32, shape=(None, None, None, self.input_depth), name='label')
            self.is_training = tf.placeholder_with_default(True, (), name='is_training')

        with tf.variable_scope('normalize_inputs') as scope:
            self.images_norm = tf.contrib.layers.batch_norm(self.images, trainable=False, epsilon=1e-6)

        with tf.variable_scope('normalize_labels') as scope:
            self.labels_norm = tf.contrib.layers.batch_norm(self.labels, trainable=False, epsilon=1e-6)
            scope.reuse_variables()
            self.label_mean = tf.get_variable('BatchNorm/moving_mean')
            self.label_variance = tf.get_variable('BatchNorm/moving_variance')
            self.label_beta = tf.get_variable('BatchNorm/beta')

    def _inference(self, X):
        for i, k in enumerate(self.filter_sizes):
            with tf.variable_scope('hidden_%i' % i) as scope:
                if i == (len(self.filter_sizes) - 1):
                    activation = None
                else:
                    activation = tf.nn.relu
                pad_amt = (k-1)/2
                X = _maybe_pad_x(X, pad_amt, self.is_training)
                X = tf.layers.conv2d(X, self.layer_sizes[i], k, activation=activation)
        return X

    def _loss(self, predictions):
        with tf.name_scope('loss'):
            err = tf.square(predictions - self.labels)
            err_filled = utils.fill_na(err, 0)
            finite_count = tf.reduce_sum(tf.cast(tf.is_finite(err), tf.float32))
            mse = tf.reduce_sum(err_filled) / finite_count
            return mse

    def _optimize(self):
        opt1 = tf.train.AdamOptimizer(self.learning_rate)
        opt2 = tf.train.AdamOptimizer(self.learning_rate*0.1)

        # compute gradients irrespective of optimizer
        grads = opt1.compute_gradients(self.loss)

        # apply gradients to first n-1 layers
        opt1_grads = [v for v in grads if 'hidden_%i' % (len(self.filter_sizes)-1) not in v[0].op.name]
        opt2_grads = [v for v in grads if 'hidden_%i' % (len(self.filter_sizes)-1) in v[0].op.name]

        self.opt = tf.group(opt1.apply_gradients(opt1_grads, global_step=self.global_step),
                            opt2.apply_gradients(opt2_grads))

    def _build_graph(self):
        self._set_placeholders()
        with tf.device(self.device):
            _prediction_norm = self._inference(self.images_norm)
            self.loss = self._loss(_prediction_norm)

        self._optimize()
        self.prediction = _prediction_norm * tf.sqrt(self.label_variance) - self.label_mean


