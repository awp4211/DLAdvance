
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SRCNN(object):

    def __init__(self, sess, image_size=33, label_size=21, batch_size=128,
                 c_dim=1, checkpoint_dir=None, sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, self.c_dim],
                                     name='images')
        self.labels = tf.placeholder(tf.float32,
                                     [None, self.label_size, self.label_size, self.c_dim],
                                     name='labels')

        self.weights = {
            'w1': tf.Variable(tf.random_normal(shape=[9, 9, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal(shape=[1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal(shape=[5, 5, 32, 1], stddev=1e-3), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        self.pred = self.model()

        # LOSS MSE
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           +self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           +self.biases['b2'])
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['w3'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           +self.biases['b3'])
        return conv3