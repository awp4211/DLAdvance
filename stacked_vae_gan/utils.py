import numpy as np
import os
import scipy
import scipy.misc
import errno

import tensorflow as tf
import tensorflow.contrib.slim as slim

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# def get_image(image_path, is_grayscale=False):
#     return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img


def inverse_transform(image):
    return (image + 1.) / 2.


def sample_label():
    num = 64
    label_vector = np.zeros((num , 128), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , (i/8)%2] = 1.0
    return label_vector


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
