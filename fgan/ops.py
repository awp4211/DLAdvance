import tensorflow as tf
import numpy as np

from utils import get_const_variable, get_rand_variable, get_dim

def linear(inputs, output_size, trainable):
    in_dim = get_dim(inputs)

    w = get_rand_variable([in_dim, output_size], 1/np.sqrt(in_dim), trainable=trainable)
    b = get_const_variable([output_size], 0.0, trainable=trainable)
    return tf.matmul(inputs, w) + b


def batch_norm(x, decay_rate=0.99, is_training=True):
    shape = x.get_shape().as_list()
    dim = shape[-1]
    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name='moments_bn')
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name='moments_bn')

    avg_mean = get_const_variable([1, dim], 0.0, False, name='avg_mean_bn')
    avg_var = get_const_variable([1, dim], 1.0, False, name='avg_var_bn')
    beta = get_const_variable([1, dim], 0.0, name='beta_bn')
    gamma = get_const_variable([1, dim], 1.0, name='gamma_bn')

    if is_training:
        avg_mean_assign_op = tf.assign(avg_mean, decay_rate*avg_mean + (1-decay_rate)*mean)
        avg_var_assign_op = tf.assign(avg_var, decay_rate*avg_var+(1-decay_rate)*var)
        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            ret = gamma * (x-mean)/ tf.sqrt(1e-6+var)+beta
    else:
        ret = gamma*(x-avg_mean) / tf.sqrt(1e-6+avg_var) + beta

    return ret


if __name__ == u'__main__':
    x = tf.placeholder(dtype = tf.float32, shape = [None, 10, 10, 3])
    batch_norm(1, x, 0.9, True)
