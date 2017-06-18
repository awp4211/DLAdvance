import h5py
import numpy as np
import scipy
import tensorflow as tf
import os
import glob

from scipy import misc
from scipy import ndimage
from PIL import Image


FLAGS = tf.app.flags.FLAGS

def read_data(path):
    """
    read h5 format data file
    :param path: file path of desired file,
            data: '.h5' file format that contains train data values
            label: '.h5' file format that contains train label values
    :return:
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def _preprocess(path, scale=3):
    """
    preprocess single image file:
        read original image as YCbCr format -> normalize
            -> apply image file with bicubic interpolation
    :param path: file path of desired file
        input_: image applied bucubic interpolation (low-resolution)
        label_: image with original resolution (high-resolution)
    :param scale:
    :return:
    """
    image = imread(path, is_grayscale=True)
    label_ = _modcrop(image, scale)

    # Must be normalized
    image = image / 255.
    label_ = label_ / 255.

    input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


def prepare_data(sess, dataset):
    """

    :param sess:
    :param dataset:
    :return:
    """
    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    return data


def imread(path, is_grayscale=True):
    """
    read image using its path
    :param path:
    :param is_grayscale:
    :return:
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def _modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no reminder
    while scaling operation.Crop image to small sub image.
    :param image:
    :param scale:
    :return:
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0: h, 0: w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0: h, 0: w]
    return image


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def input_setup(sess, config):
    nx = 0
    ny = 0
    if config.is_train:
        data = prepare_data(sess, dataset="Train")
    else:
        data = prepare_data(sess, dataset="Test")

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2 #6

    if config.is_train:
        for i in xrange(len(data)):
            _input, _label = _preprocess(data[i], config.scale)

            if len(_input.shape) == 3:
                h, w, _ = _input.shape
            else:
                h, w = _input.shape

            for x in range(0, h-config.image_size+1, config.stride):
                for y in range(0, w-config.image_size+1, config.stride):
                    sub_input = _input[x:x+config.image_size, y:y+config.image_size]# 33*33
                    sub_label = _label[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] #21*21

                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)
    else:
        _input, _label = _preprocess(data[2], config.scale)

        if len(_input.shape) == 3:
            h, w, _ = _input.shape
        else:
            h, w = _input.shape

        nx = ny = 0
        for x in range(0, h-config.image_size+1, config.stride):
            nx += 1; ny = 0
            for y in range(0, w-config.image_size+1, config.stride):
                ny += 1
                sub_input = _input[x:x+config.image_size, y:y+config.image_size]
                sub_label = _label[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size]
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    make_data(sess, arrdata, arrlabel)

    if not config.is_train:
        return nx, ny


def make_data(sess, data, label):
    """
    Make input data as h5 file format
    :param sess:
    :param data:
    :param label:
    :return:
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def merge(images, size):
    """
    merge subimages to a full image
    :param images:
    :param size:
    :return:
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img
