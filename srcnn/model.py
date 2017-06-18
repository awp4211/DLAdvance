
import time
import os
import tensorflow as tf

from utils import (
    read_data,
    input_setup,
    imsave,
    merge
)

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
        print("image shape = {0}".format(self.images.get_shape()))#(?,33,33,1)
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           + self.biases['b1'])
        print("conv1 shape = {0}".format(conv1.get_shape()))#(?,25,25,64)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           + self.biases['b2'])
        print("conv2 shape = {0}".format(conv2.get_shape()))#(?,25,25,32)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['w3'],
                                        strides=[1, 1, 1, 1], padding='VALID')
                           + self.biases['b3'])
        print("conv3 shape = {0}".format(conv3.get_shape()))#(?,21,21,1)
        return conv3

    def train(self, config):
        nx = 0
        ny = 0
        if config.is_train:
            input_setup(self.sess, config)
        else:
            nx, ny = input_setup(self.sess, config)

        if config.is_train:
            data_dir = os.path.join("./{}".format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join("./{}".format(config.checkpoint_dir), "test.h5")

        train_data, train_label = read_data(data_dir)
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load Success")
        else:
            print(" [!] Load failed")

        if config.is_train:
            print("Training .......")
            for epoch in xrange(config.epoch):
                batch_idxs = len(train_data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_images = train_data[idx*config.batch_size: (idx+1)*config.batch_size]
                    batch_labels = train_label[idx*config.batch_size: (idx+1)*config.batch_size]

                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={
                        self.images: batch_images,
                        self.labels: batch_labels
                    })

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]"
                              % ((epoch+1), counter, time.time()-start_time, err))
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        else:
            print("Testing ......")
            result = self.pred.eval({self.images: train_data, self.labels: train_label})
            result = merge(result, [nx, ny])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            result *= 255.
            imsave(result, image_path)

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints ...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
