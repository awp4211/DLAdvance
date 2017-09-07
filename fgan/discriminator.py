import tensorflow as tf
import numpy as np

from ops import linear
from layers import Layers


class Discriminator(Layers):
    """
    Discriminator has three linear layers with exponential linear unit.
    The final activation is specific to each divergence and listed in Paper.
    """
    def __init__(self, name_scopes, layer_channels,
                 input_height, input_width, input_channels,
                 f_divergence='pearson',
                 output_dim=1):
        assert(len(name_scopes) == 4)
        assert(len(layer_channels) == 3)

        super(Discriminator, self).__init__(name_scopes)
        self.layer_channels = layer_channels

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.f_divergence = f_divergence

    def set_model(self, x, is_training=True, reuse=False):
        with tf.variable_scope('reshape'):
            h = tf.reshape(x, [-1, self.input_height*self.input_width*self.input_channels])

        with tf.variable_scope(self.name_scopes[0], reuse=reuse):
            out_dim = self.layer_channels[0]
            h = linear(h, output_size=out_dim, trainable=is_training)
            h = tf.nn.elu(h)

        with tf.variable_scope(self.name_scopes[1], reuse=reuse):
            out_dim = self.layer_channels[1]
            h = linear(h, output_size=out_dim, trainable=is_training)
            h = tf.nn.elu(h)

        with tf.variable_scope(self.name_scopes[2], reuse=reuse):
            out_dim = self.layer_channels[2]
            h = linear(h, output_size=out_dim, trainable=is_training)
            if self.f_divergence == 'pearson':
                h = h
            elif self.f_divergence == 'kl':
                h = h
            elif self.f_divergence == 'rkl':
                h = -tf.exp(-h)
            elif self.f_divergence == 'squared_hellinger':
                h = tf.ones_like(h) - tf.exp(-h)
            elif self.f_divergence == 'jensen_shannon':
                h = tf.log(tf.multiply(tf.constant(2)), tf.ones_like(h)) - tf.log(tf.ones_like(h)+tf.exp(-h))
            elif self.f_divergence == 'original_gan':
                h = -tf.log(tf.ones_like(h) + tf.exp(-h))

        with tf.variable_scope(self.name_scopes[3], reuse=reuse):
            out_dim = self.output_dim
            h = linear(h, output_size=out_dim, trainable=is_training)

        return h


if __name__ == '__main__':

    input_height=28
    input_width=28
    input_channels=1

    dis = Discriminator(['d/linear1', 'd/linear2', 'd/linear3', 'd/linear4'],
                        [500, 500, 500],
                        input_height, input_width, input_channels,
                        f_divergence='pearson',
                        output_dim=1)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    h = dis.set_model(x)
    for var in dis.get_variables():
        print var.name



