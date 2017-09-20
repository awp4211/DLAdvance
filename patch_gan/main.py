import os
import numpy as np
import tensorflow as tf

#from model_v1 import PatchGAN
from model_v4 import PatchGAN
from utils import pp, visualize, show_all_variables


flags = tf.app.flags

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_integer("g_epoch", 10, "Iteration of training G when training D one time")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [28]")
flags.DEFINE_integer("input_width", 32, "The size of image to use (will be center cropped). If None, same value as input_height [28]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 32, "The size of the output images to produce. If None, same value as output_height [None]")
#flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
#flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
#flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

flags.DEFINE_integer("z_dim", 100, "noise dimension")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        patchGan = PatchGAN(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.batch_size,
                    y_dim=10,
                    z_dim=FLAGS.z_dim,
                    checkpoint_dir=FLAGS.checkpoint_dir)

        show_all_variables()

        if FLAGS.train:
            patchGan.train(FLAGS)
        else:
            if not patchGan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")


        OPTION = 1
        visualize(sess, patchGan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()