"""
paper: Unsupervised representation Learning with
Deep Convolutional Generative Adversarial Networks
arXiv :1511.06434v2
for MNIST dataset
"""

import os
import tensorflow as tf
import numpy as np
import time

from tqdm import tqdm

from ops import BatchNorm, summarys, linear, conv2d, deconv2d, conv_cond_concat, lrelu
from utils import conv_out_size_same, save_images, image_manifold_size

class DCGAN(object):

    def __init__(self, sess, input_height, input_width, batch_size=64,
                 sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, df_dim=64, gf_dim=64,
                 gfc_dim=1024, dfc_dim=1024, checkpoint_dir=None, sample_dir=None):
        """
        
        :param sess: Tensorflow session 
        :param input_height: 
        :param input_width: 
        :param batch_size: 
        :param sample_num: 
        :param output_height: 
        :param output_width: 
        :param y_dim: Dimension of dim for y.
        :param z_dim: Dimension of dim for Z.
        :param df_dim: Dimension of discriminator filters in first conv layer.
        :param gf_dim: Dimension of gen filters in first conv layer.
        :param gfc_dim: Dimension pf gen unit filters fully connected layer.
        :param dfc_dim: Dimension of discriminator unit filters fully connected layer.
        :param checkpoint_dir: 
        :param sample_dir: 
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization
        self.d_bn1 = BatchNorm(name='d_bn1')
        self.d_bn2 = BatchNorm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = BatchNorm(name='d_bn3')

        self.g_bn0 = BatchNorm(name='g_bn0')
        self.g_bn1 = BatchNorm(name='g_bn1')
        self.g_bn2 = BatchNorm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = BatchNorm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.data_x, self.data_y = self.load_mnist()
        self.c_dim = 1 # color channels of image
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        image_dim = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dim, name='real_image')
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = summarys['histogram']('z', self.z)

        # build model
        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        # build summary
        self.d_sum = summarys['histogram']('d', self.D)
        self.d__sum = summarys['histogram']("d_", self.D_)
        self.G_sum = summarys['image']("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = summarys['scalar']('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = summarys['scalar']('d_loss_fake', self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = summarys['scalar']('g_loss', self.g_loss)
        self.d_loss_sum = summarys['scalar']('d_loss', self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def generator(self, z, y=None):
        with tf.variable_scope('generator') as scope:
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_h4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project z and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
                # reshape vector to matrix
                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim*8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = \
                    deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = \
                    deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(self.h2))

                h3, self.h3_w, self.h3_b = \
                    deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(self.h3))

                h4 = self.h4_w, self.h4_b = \
                    deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # tf.expand_dims(tf.expand_dims(y, 1), 2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim+self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim+self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3), h3


    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                           [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = tf.concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')
        from tensorflow.examples.tutorials.mnist import input_data
        mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')
        mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

        data_x, data_y= mnist.train.next_batch(50000)

        data_x = np.reshape(data_x, [50000, 28, 28, 1])
        return data_x, data_y


    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).\
            minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).\
            minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        self.g_sum = summarys['merge']([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = summarys['merge']([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = summarys['writer']('.logs', self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        sample_inputs = self.data_x[0: self.sample_num]
        sample_labels = self.data_y[0: self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print "[*] load success"
        else:
            print "[!] load field !!"

        for epoch in tqdm(range(config.epoch)):
            batch_idxs = min(len(self.data_x), config.train_size) // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images = self.data_x[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_images,
                                                          self.y: batch_labels,
                                                          self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                for i in range(2):
                    # update G network twice
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z,
                                                              self.y: batch_labels})
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y: batch_labels})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.y: batch_labels})
                errG = self.g_loss.eval({self.z: batch_z, self.y: batch_labels})

                counter += 1
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                            feed_dict={self.z: batch_z,
                                                                       self.inputs: sample_inputs,
                                                                       self.y: sample_labels})
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, batch_idxs,
                     time.time() - start_time, errD_fake + errD_real, errG))



    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            "MNIST", self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        import re
        print("[*] reading checkpoints ...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0