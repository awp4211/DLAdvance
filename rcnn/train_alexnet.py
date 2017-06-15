"""
Train alexnet with 17 flowers
"""

import pickle
import numpy as np
import os
import tflearn

from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import preprocessing_RCNN as prep


def load_data(datafile, num_class, save=False, save_path='dataset.pkl'):
    """
    load 17 flowers dataset
    :param datafile:
    :param num_class:
    :param save:
    :param save_path:
    :return:
    """
    train_list = open(datafile, 'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        filepath = tmp[0]
        print(filepath)
        img = Image.open(filepath)
        img = prep.resize_image(img, 224, 224)
        np_img = prep.pil_to_nparray(img)
        images.append(np_img)

        # one-hot encoder
        index = int(tmp[1])
        label = np.zeros(num_class)
        label[index] = 1
        labels.append(label)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def create_alexnet(num_classes):
    """
    Building Alexnet
    :param num_classes:
    :return:
    """
    network = input_data(shape=[None, 224, 224, 3])
    print('network shape = %s' % network.get_shape())
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    print('conv1 shape = %s' % network.get_shape())

    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    print('conv2 shape = %s' % network.get_shape())

    network = conv_2d(network, 384, 3, activation='relu')
    print('conv3 shape = %s' % network.get_shape())

    network = conv_2d(network, 384, 3, activation='relu')
    print('conv4 shape = %s' % network.get_shape())

    network = conv_2d(network, 256, 3, activation='relu')
    print('conv5 shape = %s' % network.get_shape())

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    print('before fc shape = %s' % network.get_shape())
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    print('fc1 shape = %s' % network.get_shape())

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    print('fc2 shape = %s' % network.get_shape())

    network = fully_connected(network, num_classes, activation='softmax')
    print('fc3 shape = %s' % network.get_shape())

    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


def train(network, X, Y):
    """
    train alexnet
    :param network:
    :param X:
    :param Y:
    :return:
    """
    model = tflearn.DNN(network,
                        checkpoint_path='model_alexnet',
                        max_checkpoints=1,
                        tensorboard_verbose=2,
                        tensorboard_dir='output')
    if os.path.isfile('model_save.model'):
        model.load('model_save.model')
    model.fit(X,
              Y,
              n_epoch=100,
              validation_set=0.1,
              shuffle=True,
              show_metric=True,
              batch_size=64,
              snapshot_step=200,
              snapshot_epoch=False,
              run_id='alexnet_oxflowers17')  # epoch = 1000
    # Save the model
    model.save('model_save.model')


def predict(network, modelfile, images):
    """
    using pre-trained model to predict image's class
    :param network:
    :param modelfile:
    :param images:
    :return:
    """
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)


if __name__ == '__main__':
    print('...... loading dataset ......')
    X, Y = load_data("train_list.txt", 17)

    print('...... create network ......')
    net = create_alexnet(17)

    print('...... training network ......')
    train(net, X, Y)
