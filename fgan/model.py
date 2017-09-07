import tensorflow as tf

from generator import Generator
from discriminator import Discriminator


class Model(object):


    def __init__(self, z_dim, batch_size,
                 image_height, image_width, image_channels,
                 lr=0.0001,
                 f_divergence='pearson'):
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.lr = lr
        self.f_divergence = f_divergence

        gen_layer = [500, 500]
        disc_layer = [500, 500, 500]

        # generator
        self.G = Generator(['g/linear1', 'g/linear2', 'g/linear3'], z_dim, gen_layer,
                             image_height, image_width, image_channels)
        # discriminator
        self.D = Discriminator(['d_linear1', 'd_linear2', 'd_linear3', 'd_linear4'], disc_layer,
                                image_height, image_width, image_channels,
                                f_divergence=f_divergence,
                                output_dim=1)

    def set_model(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.image_height*self.image_width*self.image_channels])

        # generator
        gen_x = self.G.set_model(self.z, is_training=True, reuse=False)
        g_logits = self.D.set_model(gen_x, is_training=True, reuse=False)
        self.g_obj = -tf.reduce_mean(self.f_star(g_logits))
        self.train_G = tf.train.AdamOptimizer(self.lr, beta1=0.5)\
            .minimize(self.g_obj, var_list=self.G.get_variables())

        # discriminator
        d_logits = self.D.set_model(self.x, is_training=True, reuse=True)
        self.d_obj = -tf.reduce_mean(d_logits) + tf.reduce_mean(self.f_star(g_logits))
        self.train_D = tf.train.AdamOptimizer(self.lr, beta1=0.5)\
            .minimize(self.d_obj, var_list=self.D.get_variables())

        # for images generation
        self.gen_images = self.G.set_model(self.z, is_training=False, reuse=True)

    def training_G(self, sess, z_list):
        _, g_obj = sess.run([self.train_G, self.g_obj],
                            feed_dict={self.z: z_list})
        return g_obj

    def training_D(self, sess, z_list, x_list):
        _, d_obj = sess.run([self.train_D, self.d_obj],
                            feed_dict={self.z: z_list,
                                       self.x: x_list})
        return d_obj


    def gen_samples(self, sess, z):
        ret = sess.run(self.gen_images,
                       feed_dict={self.z: z})
        return ret

    def f_star(self, logits):
        if self.f_divergence == 'pearson':
            # pearson:1/4 t^2 + t
            return 0.25*tf.square(logits) + logits
        elif self.f_divergence == 'kl':
            # KL:\exp(t-1)
            return tf.exp(logits-1)
        elif self.f_divergence == 'rkl':
            # RKL:-1-\log(-t)
            return tf.subscribe(tf.ones_like(logits) - logits)
        elif self.f_divergence == 'squared_hellinger':
            # squared_hellinger:t / (1/t)
            return tf.divide(logits, tf.ones_like(logits) - logits)
        elif self.f_divergence == 'jensen_shannon':
            # -\log(2-\exp(t))
            return -tf.log(2*tf.ones_like(logits) - tf.exp(logits))
        elif self.f_divergence == 'original_gan':
            # -\log(1-\exp(t))
            return -tf.log(tf.ones_like(logits) - tf.exp(logits))


if __name__ == '__main__':
    image_height, image_width, image_channels = 28, 28, 1
    z_dim=100
    batch_size=200
    model = Model(z_dim, batch_size,
                  image_height, image_width, image_channels)
    model.set_model()