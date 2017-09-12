import os, sys
import glob
import random

from tensorflow.python.platform import gfile
from six.moves import cPickle as pickle

DATA_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip'
random.seed(5)


class CelebA_Dataset():
    def __init__(self, dict):
        self.train_images = dict['train']
        self.test_images = dict['test']
        self.validation_images = dict['validation']


def read_dataset(data_dir):
    pickle_filename = 'celebA.pickle'
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        celebA_folder = './'
        dir_path = os.path.join(data_dir, celebA_folder)
        #print dir_path
        result = create_image_lists(dir_path)
        print "*** Training set : %d " % len(result['train'])
        print "*** Test set : %d " % len(result['test'])
        print "*** Validation set : %d " % len(result['validation'])
        print "*** Pickling ***"
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print "*** Found pickle file ***"

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        celebA = CelebA_Dataset(result)
        del result
    return  celebA


def create_image_lists(image_dir, testing_percentage=0.0,
                       validation_percentage=0.0):
    if not gfile.Exists(image_dir):
        print "Image directory '" + image_dir +"' not found."
        return None

    training_images = []
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []

    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.'+extension)
        file_list.extend(glob.glob(file_glob))

    if not file_list:
        print "No files found"
    else:
        training_images.extend([f for f in file_list])

    random.shuffle(training_images)
    no_of_images = len(training_images)
    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    result = {
        'train': training_images,
        'test': testing_images,
        'validation': validation_images,
    }
    return result


if __name__ == '__main__':
    image_dir = 'img_align_celeba'
    result = create_image_lists(image_dir)
    training_dataset = result['train']
    #print training_dataset
    print len(training_dataset)

    read_dataset(image_dir)

