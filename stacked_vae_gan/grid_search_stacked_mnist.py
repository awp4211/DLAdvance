import tensorflow as tf
import os

from utils import mkdir_p
from stacked_vaegan_mnist import stacked_vaegan_mnist

from data_mnist import load_mnist_32

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("max_epoch", 20, "the maxmization epoch")
flags.DEFINE_string("data_path", "MNIST_DATA", "MNIST dataset path")
flags.DEFINE_string("gpu", "0", "use %no gpu to run")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    data_x, _ = load_mnist_32(FLAGS.data_path)
    print "the num of dataset %d" % data_x.shape[0]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    for learning_rate in [1E-4, 1E-5, 1E-6]:
        for latent_dim in [20, 50, 100, 128]:
            for clipped_gradient in [True, False]:
                for num_vae in [2, 3, 4]:

                    tf.reset_default_graph()

                    root_log_dir = "./runtime/logs/mnist_vaegan/lr_%s_z_%d_gp_%s_nvae_%d" % (learning_rate, latent_dim, clipped_gradient, num_vae)
                    vaegan_checkpoint_dir = "./runtime/models/mnist_vaegan/lr_%s_z_%d_gp_%s_nvae_%d/mnist_model.ckpt" % (learning_rate, latent_dim, clipped_gradient, num_vae)
                    sample_path = "./runtime/samples/mnist_vaegan/lr_%s_z_%d_gp_%s_nvae_%d" % (learning_rate, latent_dim, clipped_gradient, num_vae)

                    mkdir_p(root_log_dir)
                    mkdir_p(vaegan_checkpoint_dir)
                    mkdir_p(sample_path)

                    model_path = vaegan_checkpoint_dir

                    batch_size = FLAGS.batch_size
                    max_epoch = FLAGS.max_epoch

                    vaeGan = stacked_vaegan_mnist(batch_size=batch_size,
                                          max_epoch=max_epoch,
                                          model_path=model_path,
                                          data=data_x,
                                          latent_dim=latent_dim,
                                          sample_path=sample_path,
                                          log_dir=root_log_dir,
                                          learnrate_init=learning_rate,
                                          num_vae=num_vae,
                                          clipped_gradient=clipped_gradient)

                    vaeGan.build_model_vaegan()
                    vaeGan.train()