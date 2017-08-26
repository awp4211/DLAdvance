import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')

mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
n_samples = mnist.train.num_examples

np.random.seed(0)
tf.set_random_seed(0)


def xavier_init(fan_in, fan_out, constant=1):
    """
    Xaiver initialization of network weights
    :param fan_in:
    :param fan_out:
    :param constant:
    :return:
    """
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """
    Variational AutoEncoder(VAE) with an sklearn-like interface implemented using Tensorflow.
    This implementation uses probabilistic encoders and decoders using Gaussian distributions
    and realized by multi-layer perceptrons.
    The VAE can be learned end-to-end.
    """