import numpy as np
import os

from tqdm import tqdm


def load_mnist_32(mnist_path="MNIST_DATA"):
    print "[x] load mnist 32 dataset "
    from skimage import transform
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), mnist_path)
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
    print "[x] load mnist success"
    return data_x, data_y


def shuffle(data):
    length = data.shape[0]
    perm = np.arange(length)
    np.random.shuffle(perm)
    return data[perm]
