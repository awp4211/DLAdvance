import os
import tensorflow as tf
import numpy as np
import time

from tqdm import tqdm

from ops import BatchNorm, summarys, linear, conv2d, deconv2d, conv_cond_concat, lrelu
from utils import save_images, image_manifold_size, generate_z


class PatchGAN(object):
    def __init__(self,
                 sess,
                 input_height, input_width,
                 batch_size,
                 sample_num,
                 output_height, output_width,
                 checkpoint_dir,
                 y_dim=10,
                 z_dim=100,
                 df_dim=64,
                 gf_dim=64,
                 gfc_dim=1024,
                 dfc_dim=1024,
                 c_dim=1,
                 num_patches=4):
        self.sess = sess
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.num_patches = num_patches

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batchnorm for patch generator
        g_bns_all = []
        for patch in range(num_patches):
            g_bns = []
            g_bn0 = BatchNorm(name="g_bn_p%d_0" % patch)
            g_bn1 = BatchNorm(name="g_bn_p%d_1" % patch)
            g_bn2 = BatchNorm(name="g_bn_p%d_2" % patch)
            g_bns.append(g_bn0)
            g_bns.append(g_bn1)
            g_bns.append(g_bn2)
            g_bns_all.append(g_bns)
        self.g_bns_all = g_bns_all

        # batchnorm for full generator
        self.g_bn_f0 = BatchNorm(name="g_bn_f_0")
        self.g_bn_f1 = BatchNorm(name="g_bn_f_1")
        self.g_bn_f2 = BatchNorm(name="g_bn_f_2")

        self.d_bn1 = BatchNorm(name="d_bn1")
        self.d_bn2 = BatchNorm(name="d_bn2")
        self.d_bn3 = BatchNorm(name="d_bn3")

        self.checkpoint_dir = checkpoint_dir
        self.data_x, self.data_y = self.load_mnist_32()
        self.build_model()

    def build_model(self):

        # set placeholders
        patch_image_dim = []

        self.x_set = []
        self.y_set = []
        self.z_set = []

        for patch in range(self.num_patches):
            x = tf.placeholder(tf.float32, [self.batch_size,
                                            int(self.input_height/np.sqrt(self.num_patches)),
                                            int(self.input_width/np.sqrt(self.num_patches)),
                                            self.c_dim ],
                               name="x_%d" % patch)
            self.x_set.append(x)

            y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim+2], name="y_%d" % patch)
            self.y_set.append(y)

            z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z_%d"%patch)
            self.z_set.append(z)

        self.g_set = []
        for patch in range(self.num_patches):
            G = self.patch_generator(index=patch, z=self.z_set[patch], y=self.y_set[patch])
            self.g_set.append(G)

        # merge z patches
        self.z = tf.concat([self.z_set[0], self.z_set[1], self.z_set[2], self.z_set[3]], axis=1)
        self.z_sum = summarys["histogram"]("z", self.z)

        # merge G patches
        g_0 = tf.pad(self.g_set[0], paddings=[[0, 0], [0, self.output_height / 2], [0, self.output_width / 2], [0, 0]], mode="CONSTANT")
        g_1 = tf.pad(self.g_set[1], paddings=[[0, 0], [self.output_height / 2, 0], [0, self.output_width / 2], [0, 0]], mode="CONSTANT")
        g_2 = tf.pad(self.g_set[2], paddings=[[0, 0], [0, self.output_height / 2], [self.output_width / 2, 0], [0, 0]], mode="CONSTANT")
        g_3 = tf.pad(self.g_set[3], paddings=[[0, 0], [self.output_height / 2, 0], [self.output_width / 2, 0], [0, 0]], mode="CONSTANT")
        self.G_patch = tf.add(tf.add(tf.add(g_0, g_1), g_2), g_3)

        self.G_f = self.generator(self.z_set, self.y_set)

        self.G = self.G_patch + self.G_f

        # merge x patches
        x_0 = tf.pad(self.x_set[0], paddings=[[0, 0], [0, self.input_height / 2], [0, self.input_width / 2], [0, 0]], mode="CONSTANT")
        x_1 = tf.pad(self.x_set[1], paddings=[[0, 0], [self.input_height / 2, 0], [0, self.input_width / 2], [0, 0]], mode="CONSTANT")
        x_2 = tf.pad(self.x_set[2], paddings=[[0, 0], [0, self.input_height / 2], [self.input_width / 2, 0], [0, 0]], mode="CONSTANT")
        x_3 = tf.pad(self.x_set[3], paddings=[[0, 0], [self.input_height / 2, 0], [self.input_width / 2, 0], [0, 0]], mode="CONSTANT")
        self.x = tf.add(tf.add(tf.add(x_0, x_1), x_2), x_3)

        # build model
        self.D, self.D_logits = self.discriminator(self.x, self.y_set, reuse=False)
        self.sampler = self.sampler(self.z_set, self.y_set)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y_set, reuse=True)

        # build summary
        self.d_sum = summarys["histogram"]("d", self.D)
        self.d__sum = summarys["histogram"]("d_", self.D_)
        self.G_sum = summarys["image"]("G", self.G)

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
        self.d_vars = [var for var in t_vars if "d_" in var.name]
        self.g_vars = [var for var in t_vars if "g_" in var.name]

        self.saver = tf.train.Saver()

    def patch_generator(self, index, z, y=None):
        """
        patch generator
        :param z: 
        :param y: 
        :param scope: 
        :return: 
        """
        with tf.variable_scope("generator_%d"%index) as scope:
            s_h, s_w = self.output_height/int(np.sqrt(self.num_patches)), \
                       int(self.output_width/np.sqrt(self.num_patches)) # 16, 16
            s_h2, s_w2 = int(s_h/2), int(s_w/2)  # 8, 8
            s_h4, s_w4 = int(s_h2/2), int(s_w2/2)  # 4, 4

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim+2])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bns_all[index][0](linear(z, self.gfc_dim, "g_h0_lin")))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bns_all[index][1](linear(h0, self.gf_dim*4*s_h4*s_w4, "g_h1_lin")))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*4])
            h1 = conv_cond_concat(h1, yb)

            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*2], name="g_h2")
            h2 = self.g_bns_all[index][2](h2)
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name="g_h3"))

    def generator(self, z_set, y_set):
        """
        Fully generator
        :param z_set: 
        :param y_set: 
        :return: 
        """
        z = tf.concat([v for v in z_set], axis=1)
        y = y_set[0]
        with tf.variable_scope("generator_f") as scope:
            s_h, s_w = self.output_height, self.output_width # 32, 32
            s_h2, s_w2 = int(s_h / 2), int(s_w / 2)  # 16, 16
            s_h4, s_w4 = int(s_h2 / 2), int(s_w2 / 2)  # 8, 8

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim + 2])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn_f0(linear(z, self.gfc_dim, "g_h0_lin")))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn_f1(linear(h0, self.gf_dim * 4 * s_h4 * s_w4, "g_h1_lin")))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 4])
            h1 = conv_cond_concat(h1, yb)

            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name="g_h2")
            h2 = self.g_bn_f2(h2)
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name="g_h3"))

    def discriminator(self, x, y_set=None, reuse=False):
        """
        D
        :param x: 
        :param y: list 
        :param reuse: 
        :return: 
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            y = tf.concat([y_set[0], y_set[1], y_set[2], y_set[3]], axis=1)
            yb = tf.reshape(y, [self.batch_size, 1, 1, (self.y_dim+2)*4])
            x = conv_cond_concat(x, yb)

            h0 = lrelu(conv2d(x, self.c_dim+self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')
            return tf.nn.sigmoid(h3), h3

    def sampler(self, z_set, y_set=None):
        """
        Sample instance from noise
        :param z_set: list, contains self.num_patches's noise vector
        :param y_set: list, contains self.num_patches's condition vector
        :return: 
        """
        s_h, s_w = self.output_height / int(np.sqrt(self.num_patches)), int(
            self.output_height / np.sqrt(self.num_patches)) # 16, 16
        s_h2, s_w2 = int(s_h / 2), int(s_w / 2)  # 8, 8
        s_h4, s_w4 = int(s_h2 / 2), int(s_w2 / 2)  # 4, 4

        g_patchs = []
        for index in range(self.num_patches):
            with tf.variable_scope("generator_%d" % index) as scope:
                scope.reuse_variables()
                yb = tf.reshape(y_set[index], [self.batch_size, 1, 1, self.y_dim+2])
                z_ = tf.concat([z_set[index], y_set[index]], 1)

                h0 = tf.nn.relu(self.g_bns_all[index][0](linear(z_, self.gfc_dim, "g_h0_lin")))
                h0 = tf.concat([h0, y_set[index]], 1)

                h1 = tf.nn.relu(self.g_bns_all[index][1](linear(h0, self.gf_dim * 4 * s_h4 * s_w4, "g_h1_lin")))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 4])
                h1 = conv_cond_concat(h1, yb)

                h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name="g_h2")
                h2 = self.g_bns_all[index][2](h2)
                h2 = tf.nn.relu(h2)
                h2 = conv_cond_concat(h2, yb)

                h3 = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name="g_h3"))
                g_patchs.append(h3)

        hh_0 = tf.pad(g_patchs[0], paddings=[[0, 0], [0, self.output_height / 2], [0, self.output_width / 2], [0, 0]], mode="CONSTANT")
        hh_1 = tf.pad(g_patchs[1], paddings=[[0, 0], [self.output_height / 2, 0], [0, self.output_width / 2], [0, 0]], mode="CONSTANT")
        hh_2 = tf.pad(g_patchs[2], paddings=[[0, 0], [0, self.output_height / 2], [self.output_width / 2, 0], [0, 0]], mode="CONSTANT")
        hh_3 = tf.pad(g_patchs[3], paddings=[[0, 0], [self.output_height / 2, 0], [self.output_width / 2, 0], [0, 0]], mode="CONSTANT")

        hh = tf.add(tf.add(tf.add(hh_0, hh_1), hh_2), hh_3)

        z = tf.concat([v for v in z_set], axis=1)
        y = y_set[0]
        with tf.variable_scope("generator_f") as scope:
            scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width  # 32, 32
            s_h2, s_w2 = int(s_h / 2), int(s_w / 2)  # 16, 16
            s_h4, s_w4 = int(s_h2 / 2), int(s_w2 / 2)  # 8, 8

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim + 2])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn_f0(linear(z, self.gfc_dim, "g_h0_lin")))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn_f1(linear(h0, self.gf_dim * 4 * s_h4 * s_w4, "g_h1_lin")))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 4])
            h1 = conv_cond_concat(h1, yb)

            h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name="g_h2")
            h2 = self.g_bn_f2(h2)
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            gf = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name="g_h3"))

        return tf.add(gf, hh)


    def load_mnist_32(self):
        print "[x] load mnist dataset "
        from skimage import transform
        from tensorflow.examples.tutorials.mnist import input_data
        mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')
        mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

        train_data_x, train_data_y = mnist.train.next_batch(55000)
        validation_data_x, validation_data_y = mnist.validation.next_batch(5000)

        data_x = np.concatenate([train_data_x, validation_data_x], axis=0)
        data_y = np.concatenate([train_data_y, validation_data_y], axis=0)

        data_x = np.reshape(data_x, [60000, 28, 28, 1])
        new_data_x = []
        for i in tqdm(range(60000)):
            new_data_x.append(transform.resize(data_x[i], (32, 32)))
        data_x = np.array(new_data_x).astype(np.float32)
        return data_x, data_y

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        self.g_sum = summarys['merge']([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = summarys['merge']([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = summarys['writer']('.logs', self.sess.graph)

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
                batch_images = self.data_x[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_z = generate_z(self.batch_size, self.z_dim*4)

                # update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={self.x_set[0]: batch_images[:, 0: int(self.input_height/np.sqrt(self.num_patches)), 0:int(self.input_width/np.sqrt(self.num_patches)), :],
                               self.x_set[1]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                               self.x_set[2]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                               self.x_set[3]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                               self.z_set[0]: batch_z[:, 0: self.z_dim],
                               self.z_set[1]: batch_z[:, self.z_dim: self.z_dim*2],
                               self.z_set[2]: batch_z[:, self.z_dim*2: self.z_dim*3],
                               self.z_set[3]: batch_z[:, self.z_dim*3: self.z_dim*4],
                               self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                               self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                               self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                               self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)
                               })
                self.writer.add_summary(summary_str, counter)

                for i in range(config.g_epoch):
                    # update G network g_epoch times
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.z_set[0]: batch_z[:, 0: self.z_dim],
                                                       self.z_set[1]: batch_z[:, self.z_dim: self.z_dim * 2],
                                                       self.z_set[2]: batch_z[:, self.z_dim * 2: self.z_dim * 3],
                                                       self.z_set[3]: batch_z[:, self.z_dim * 3: self.z_dim * 4],
                                                       self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                       self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                                                       self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                       self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)})
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z_set[0]: batch_z[:, 0: self.z_dim],
                                                   self.z_set[1]: batch_z[:, self.z_dim: self.z_dim*2],
                                                   self.z_set[2]: batch_z[:, self.z_dim*2: self.z_dim*3],
                                                   self.z_set[3]: batch_z[:, self.z_dim*3: self.z_dim*4],
                                                   self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)
                                                   })
                errD_real = self.d_loss_real.eval({self.x_set[0]: batch_images[:, 0: int(self.input_height/np.sqrt(self.num_patches)), 0:int(self.input_width/np.sqrt(self.num_patches)), :],
                                                   self.x_set[1]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                   self.x_set[2]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                   self.x_set[3]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                   self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                   self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)
                                                   })
                errG = self.g_loss.eval({self.z_set[0]: batch_z[:, 0: self.z_dim],
                                         self.z_set[1]: batch_z[:, self.z_dim: self.z_dim*2],
                                         self.z_set[2]: batch_z[:, self.z_dim*2: self.z_dim*3],
                                         self.z_set[3]: batch_z[:, self.z_dim*3: self.z_dim*4],
                                         self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                         self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                                         self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                         self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)
                                         })

                counter += 1
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                            feed_dict={
                                                                self.x_set[0]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                                self.x_set[1]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                                self.x_set[2]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                                self.x_set[3]: batch_images[:, 0: int(self.input_height / np.sqrt(self.num_patches)), 0:int(self.input_width / np.sqrt(self.num_patches)), :],
                                                                self.z_set[0]: batch_z[:, 0: self.z_dim],
                                                                self.z_set[1]: batch_z[:, self.z_dim: self.z_dim * 2],
                                                                self.z_set[2]: batch_z[:, self.z_dim * 2: self.z_dim * 3],
                                                                self.z_set[3]: batch_z[:, self.z_dim * 3: self.z_dim * 4],
                                                                self.y_set[0]: np.concatenate([batch_labels, np.array([[0, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                                self.y_set[1]: np.concatenate([batch_labels, np.array([[0, 1]] * self.batch_size).astype(np.float32)], axis=1),
                                                                self.y_set[2]: np.concatenate([batch_labels, np.array([[1, 0]] * self.batch_size).astype(np.float32)], axis=1),
                                                                self.y_set[3]: np.concatenate([batch_labels, np.array([[1, 1]] * self.batch_size).astype(np.float32)], axis=1)
                                                            })
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

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