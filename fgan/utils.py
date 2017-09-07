import tensorflow as tf
import numpy as np
import scipy.misc
import math

def get_rand_variable(shape, stddev, trainable=True, name=None):
    return tf.get_variable(name if name is not None else 'weights',
                           shape,
                           initializer=tf.random_normal_initializer(stddev=stddev),
                           trainable=trainable)


def get_const_variable(shape, value, trainable=True, name=None):
    return tf.get_variable(name if name is not None else 'biases',
                           shape,
                           initializer=tf.constant_initializer(value),
                           trainable=trainable)


def shuffle(x, nrr):
    """
    shuffle data
    :param x: 
    :param nrr: np.random.RandomState()
    :return: 
    """
    rand_ix = nrr.permutation(x.shape[0])
    return x[rand_ix]


def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim*=d
    return dim

def save_images(images, img_shape, img_name):
    """
    merge images and save 
    :param images: 
    :param img_shape: 
    :param img_path: 
    :param img_name: 
    :return: 
    """
    h, w = img_shape[0], img_shape[1]
    image_frame_dim = int(math.ceil(images.shape[0] ** .5))
    img = np.zeros((h * image_frame_dim, w * image_frame_dim, 1))
    for idx, image in enumerate(images):
        #image = np.squeeze(image)
        i = idx % image_frame_dim
        j = idx // image_frame_dim
        img[j * h:j * h + h, i * w:i * w + w, :] = image*255.0

    img = np.squeeze(img)
    scipy.misc.imsave(img_name, img)

