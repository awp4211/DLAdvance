from __future__ import print_function

__author__ = "shekkizh"
"""
Tensorflow implementation of Wasserstein GAN
"""
import numpy as np
import tensorflow as tf
from models.GAN_models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/CelebA_faces/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_integer("model", "0", "Model to train. 0 - GAN, 1 - WassersteinGAN")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
tf.flags.DEFINE_string("mode", "train", "train / visualize model")
tf.flags.DEFINE_integer("nr_iter_vis", 1000, "number of iterations during visualization")
tf.flags.DEFINE_integer("nr_batches_tsne", 100, "number of batches to run for tsne visualization")
tf.flags.DEFINE_bool("plot_iter_error", True, "plot the iterations error during visualization")


def main(argv=None):
    gen_dim = FLAGS.gen_dimension
    generator_dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 3]
    discriminator_dims = [3, 64, 64 * 2, 64 * 4, 64 * 8, 1]

    crop_image_size, resized_image_size = map(int, FLAGS.image_size.split(','))
    trainable_z = False
    trainable_image = False
    if FLAGS.mode in ("z_iterator_visualize", "z_iterator_tsne"):
        trainable_z = True
    if FLAGS.mode in ("image_iterator_visualize"):
        trainable_image = True
    if FLAGS.model == 0:
        model = GAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir)
    elif FLAGS.model == 1:
        model = WasserstienGAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
                               clip_values=(-0.01, 0.01), critic_iterations=5)
    else:
        raise ValueError("Unknown model identifier - FLAGS.model=%d" % FLAGS.model)

    model.create_network(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param, trainable_z=trainable_z, trainable_image=trainable_image)

    model.initialize_network(FLAGS.logs_dir)

    if FLAGS.mode == "train":
        model.train_model(int(1 + FLAGS.iterations))
    elif FLAGS.mode == "visualize":
        model.visualize_model()
    elif FLAGS.mode == "image_iterator_visualize":
        model.image_iterator_visualize_model(nr_iterations=FLAGS.nr_iter_vis, plot_iteration_error=FLAGS.plot_iter_error)
    elif FLAGS.mode == "z_iterator_visualize":
        model.z_iterator_visualize_model(nr_iterations=FLAGS.nr_iter_vis, plot_iteration_error=FLAGS.plot_iter_error)
    elif FLAGS.mode == "z_iterator_tsne":
        model.z_iterator_tsne_model(nr_iterations=FLAGS.nr_iter_vis, nr_batches_tsne=FLAGS.nr_batches_tsne)


if __name__ == "__main__":
    tf.app.run()
