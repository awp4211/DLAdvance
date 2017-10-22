import os
import tensorflow as tf


from PIL import Image
from tqdm import tqdm


OUTPUT_SIZE = 128
DEPTH = 3


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    tf.train.Feature()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(data_path, tfrecord_path):
    """
    Converts a dataset to tfrecords
    :param data_path:
    :param name:
    :return:
    """
    rows = 64
    cols = 64
    depth = DEPTH
    #
    for i in tqdm(range(12)):
        writer = tf.python_io.TFRecordWriter(tfrecord_path + str(i) + ".tfrecords")
        for img_name in os.listdir(data_path)[i*16384: (i+1)*16384]:
            img_path = data_path + img_name
            img = Image.open(img_path)

            # clip
            h, w = img.size[:2]
            j, k = (h - OUTPUT_SIZE) / 2, (w - OUTPUT_SIZE) / 2
            box = (j, k, j + OUTPUT_SIZE, k + OUTPUT_SIZE)
            img = img.crop(box=box)
            img = img.resize((rows, cols))

            # to bytes
            img_raw = img.tobytes()
            # to example
            example = tf.train.Example(features=tf.train.Feature(feature={
                "height": _int64_feature(rows),
                "width": _int64_feature(cols),
                "depth": _int64_feature(depth),
                "image_raw": _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    current_dir = os.getcwd()
    data_path = current_dir + "/data/img_align_celeba/"
    tfrecord_path = current_dir + "/data/img_align_celeba_tfrecords/train"
    os.makedirs(tfrecord_path)
    convert_to_tfrecord(data_path, tfrecord_path)