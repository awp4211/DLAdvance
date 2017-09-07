import tensorflow as tf

from ops import linear, batch_norm
from layers import Layers

class Generator(Layers):
    """
    Generate images for MNIST data which contains two linear layers each followed by
        batch normalization and ReLU activation and a final linear layer followed by
        sigmoid function.
    """
    def __init__(self, name_scopes, in_dim, layer_channels,
                 output_height, output_width, output_channels):
        assert(len(name_scopes) == 3)
        assert(len(layer_channels) == 2)
        super(Generator, self).__init__(name_scopes)
        self.in_dim = in_dim
        self.layer_channels = layer_channels

        self.output_height = output_height
        self.output_width = output_width
        self.output_channels = output_channels

    def set_model(self, z, is_training=True, reuse=False):
        with tf.variable_scope(self.name_scopes[0], reuse=reuse):
            out_dim = self.layer_channels[0]
            h = linear(z, output_size=out_dim,trainable=is_training)
            h = batch_norm(h, decay_rate=0.99, is_training=is_training)
            h = tf.nn.relu(h)

        with tf.variable_scope(self.name_scopes[1], reuse=reuse):
            out_dim = self.layer_channels[1]
            h = linear(h, output_size=out_dim, trainable=is_training)
            h = batch_norm(h, decay_rate=0.99, is_training=is_training)
            h = tf.nn.relu(h)

        with tf.variable_scope(self.name_scopes[2], reuse=reuse):
            out_dim = self.output_height*self.output_width*self.output_channels
            h = linear(h, output_size=out_dim, trainable=is_training)
            h = batch_norm(h, decay_rate=0.99, is_training=is_training)
            h = tf.nn.sigmoid(h)

        with tf.name_scope('reshape'):
            h = tf.reshape(h, [-1, self.output_height, self.output_width, self.output_channels])
        return h


if __name__ == '__main__':
    z_dim = 100
    output_height=28
    output_width=28
    output_channels=1
    g = Generator(['g/linear1', 'g/linear2', 'g/linear3'], z_dim, [500, 500],
                  output_height, output_width, output_channels)

    z = tf.placeholder(tf.float32, [None, z_dim])
    h = g.set_model(z)

    for var in g.get_variables():
        print var.name

    print h.get_shape()

    print h

