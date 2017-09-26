import os
import sys
import numpy as np
import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def get_weight_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_bias_varible(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def conv2d(layer_name, x, filter_shape):
    with tf.variable_scope(layer_name):
        w = get_weight_varible('w', filter_shape)
        b = get_bias_varible('b', filter_shape[-1])
        y = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME'), b)
        return y


def pool2d(layer_name, x):
    with tf.variable_scope(layer_name):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y


def fc(layer_name, x, inp_shape, out_shape):
    with tf.variable_scope(layer_name):
        inp_dim = inp_shape[-1]
        out_dim = out_shape[-1]
        y = tf.reshape(x, shape=inp_shape)
        w = get_weight_varible('w', [inp_dim, out_dim])
        b = get_bias_varible('b', [out_dim])
        y = tf.add(tf.matmul(y, w), b)
        return y


def build_model(x):
    y = tf.reshape(x,shape=[-1, 28, 28, 1])
    #layer 1
    y = conv2d('conv_1', y, [3, 3, 1, 8])
    y = pool2d('pool_1', y)
    #layer 2
    y = conv2d('conv_2', y, [3, 3, 8, 16])
    y = pool2d('pool_2', y)
    #layer fc
    y = fc('fc', y, [-1, 7*7*16], [-1, 10])
    return y



def average_losses(loss):
    tf.add_to_collection("losses", loss)

    # assemble all of the losses for the current tower only
    losses = tf.get_collection("losses")

    # calculate the total loss for the current tower
    regularization


