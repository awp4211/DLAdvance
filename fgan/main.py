import os
import sys
import tensorflow as tf
import numpy as np

from model import Model
from utils import shuffle, save_images
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    # parameter
    epoch_num = 100000

    gen_dir = 'gen_result'
    if not os.path.isdir(gen_dir):
        os.mkdir(gen_dir)

    image_height, image_width, image_channels = 28, 28, 1
    z_dim = 100
    batch_size =64
    f_divergence = 'person'
    assert(f_divergence in ['pearson', 'kl', 'rkl',
                            'squared_hellinger', 'jensen_shannon',
                            'original_gan'])

    # make model
    print '====== make model ======'
    model = Model(z_dim, batch_size,
                  image_height, image_width, image_channels,
                  f_divergence='person')
    model.set_model()

    # get data
    mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')
    print 'loading data from %s ' % mnist_data_path
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
    print 'loading data done'
    nrr = np.random.RandomState()

    data_x = np.concatenate([mnist.train.images, mnist.validation.images], axis=0)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):
            dataset_x = shuffle(data_x, nrr)
            g_obj = 0.0
            d_obj = 0.0
            for mini_batch in range(data_x.shape[0] // batch_size):
                batch_x = dataset_x[mini_batch*batch_size:(mini_batch+1)*batch_size]
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

                d_obj += model.training_D(sess, batch_z, batch_x)
                g_obj += model.training_G(sess, batch_z)

            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj / (data_x.shape[0]//batch_size),
                                                            g_obj / (data_x.shape[0]//batch_size)))
            save_images(images=model.gen_samples(sess, np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)),
                        img_shape=[image_height, image_width],
                        img_name=os.path.join(gen_dir, 'epoch_%d.jpg' % epoch))

