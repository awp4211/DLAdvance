"""
This is an example of distribution approximation using GAN in tensorflow
"""

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from tqdm import tqdm

sns.set(color_codes=True)

seed = 42

np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):

    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w',
                            [input.get_shape()[1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b',
                            [output_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))
    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def log(x):
    """
    Sometimes discriminator outputs can reach values close to zero due to numerical rounding,'
    This just makes sure that we exclude thoise values so that we don't end up with NaNs during optimisation
    """
    return tf.log(tf.maximum(x, 1e-5))

def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step ,var_list=var_list)
    return optimizer


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise distribution as input
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from the true data distribution
        #  self.x and the generated samples self.z
        # Here we create two copies of the discriminator network that share parameters, as you cannot use
        #  the same network with different inputs in Tensorflow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x, params.hidden_size, params.minibatch)
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, params.hidden_size, params.minibatch)

        # Define the loss for discriminator and generator networks and create optimizers for both.(Minimize)
        self.loss_d = tf.reduce_mean(- log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(- log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def samples(model, session, data, sample_range, batch_size, num_points=10000, num_bins=100):
    """
    Return a tuple(db, pd, pg) where 
        db is the current decision boundary,
        pd is a histogram of samples from the data distribution,
        pg is a histogram of generated samples.
    """
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i: batch_size * (i + 1)] = session.run(model.D1,
                                                               feed_dict={model.x: np.reshape(
                                                                   xs[batch_size * i:batch_size * (i + 1)],
                                                                   (batch_size, 1))}
                                                               )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generate samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(model.G,
                                                             feed_dict={model.z: np.reshape(
                                                                 zs[batch_size * i:batch_size * (i + 1)],
                                                                 (batch_size, 1))})

    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_plot_distributions(samps, sample_range, filename):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in tqdm(range(params.num_steps + 1)):
            # update discriminator
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _ = session.run([model.loss_d, model.opt_d],
                                    feed_dict={model.x: np.reshape(x, (params.batch_size, 1)),
                                               model.z: np.reshape(z, (params.batch_size, 1))})

            # update generator
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g],
                                    feed_dict={model.z: np.reshape(z, (params.batch_size, 1))})

            """
            if step % params.log_every == 0:
                print "{}:{:.4f}\t{:.4f}".format(step, loss_d, loss_g)
            """

            if step % params.anim_every == 0:
                samps = samples(model, session, data, gen.range, params.batch_size)
                save_plot_distributions(samps, gen.range, params.anim_path + '%d.jpg' % step)
                anim_frames.append(samps)


def model(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)


def main(args):
    model = GAN(args)
    if not os.path.isdir(args.anim_path): os.mkdir(args.anim_path)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=5000,
        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default='anim/',
        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=50,
        help='save every Nth frame for animation')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())