import tensorflow as tf
import os

from utils import mkdir_p
from vaegan_celeba import vaegan
from data_celebA import CelebA

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("max_epoch", 60, "the maxmization epoch")
flags.DEFINE_integer("latent_dim", 128, "the dim of latent code")
flags.DEFINE_integer("learn_rate_init", 0.0003, "the init of learn rate")
flags.DEFINE_string("path", "celeba", "the dataset directory")
flags.DEFINE_integer("operation", 0, "the init of learn rate")

flags.DEFINE_string("data_path", "MNIST_DATA", "MNIST dataset path")
flags.DEFINE_string("gpu", "0", "use %no gpu to run")


FLAGS = flags.FLAGS

if __name__ == "__main__":

    root_log_dir = "./logs/celeba_vaegan"
    vaegan_checkpoint_dir = "./models/celeba_vaegan/celeba_model.ckpt"
    sample_path = "./samples/celeba_vaegan"

    mkdir_p(root_log_dir)
    mkdir_p(vaegan_checkpoint_dir)
    mkdir_p(sample_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_epoch = FLAGS.max_epoch
    latent_dim = FLAGS.latent_dim

    learn_rate_init = FLAGS.learn_rate_init

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    data_list = CelebA().load_celebA(image_path=FLAGS.path)
    print "the num of dataset", len(data_list)

    vaeGan = vaegan(batch_size=batch_size,
                    max_epoch=max_epoch,
                    model_path=model_path,
                    data=data_list,
                    latent_dim=latent_dim,
                    sample_path=sample_path,
                    log_dir=root_log_dir,
                    learnrate_init=learn_rate_init)

    if FLAGS.operation == 0:

        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:

        vaeGan.build_model_vaegan()
        vaeGan.test()
