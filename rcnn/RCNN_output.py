import pickle
import numpy as np
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.io as skimage_io
import os
import tflearn


from PIL import Image
from sklearn import svm


import preprocessing_RCNN as prep
from fine_tune_RCNN import create_alexnet


def image_proposal(img_path):
    """
    using selective search to generate proposals of image
    :param img_path:
    :return:
    """
    img = skimage_io.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle
        if r['rect'] in candidates:
            continue
        if r['size']<220:
            continue
        # resize to 224 * 224 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # delete empty
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w==0 or h==0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a==0 or b==0 or c==0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = prep.resize_image(im, 224, 224)
        candidates.add(r['rect'])
        img_float = prep.pil_to_nparray(resized_proposal_img)
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


def generate_single_svm_train(one_class_train_file):
    """
    generate one class's image's proposals
    :param one_class_train_file:
    :return:
    """
    trainfile = one_class_train_file
    savepath = one_class_train_file.replace('txt', 'pkl')
    images = []
    Y = []
    if os.path.isfile(savepath):
        print("restoring svm dataset " + savepath)
        images, Y = prep.load_from_pkl(savepath)
    else:
        print("loading svm dataset " + savepath)
        images, Y = prep.load_train_proposals(trainfile,
                                              num_class=2,
                                              threshold=0.3, svm=True,
                                              save=True, save_path=savepath)
    return images, Y


def train_svms(train_file_folder, model):
    """
    using svm_train folder's file
    :param train_file_folder:
    :param model:
    :return:
    """
    listings = os.listdir(train_file_folder)
    svms = []
    for train_file in listings:
        if "pkl" in train_file:
            continue
        X, Y = generate_single_svm_train(train_file_folder+train_file)
        train_features = []
        for i in X:
            feats = model.predict([i])
            train_features.append(feats[0])
        print("feature dimension")
        print(np.shape(train_features))
        clf = svm.LinearSVC()
        print("fit svm")
        clf.fit(train_features, Y)
        svms.append(clf)
    return svms


if __name__ == '__main__':
    train_file_folder = 'svm_train/'
    img_path = '2flowers/jpg/0/image_0561.jpg'
    imgs, verts = image_proposal(img_path)
    net = create_alexnet(3)
    model = tflearn.DNN(net)
    model.load('fine_tune_model_save.model')
    svms = train_svms(train_file_folder, model)
    print("Done fitting svms")
    features = model.predict(imgs)
    print("predict image:")
    print(np.shape(features))

    results = []
    results_label = []
    count = 0

    for f in features:
        for i in svms:
            pred = i.predict(f)
            print(pred)
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count += 1

    print("result:")
    print(results)
    print("result label:")
    print(results_label)

    img = skimage_io.imread(img_path)
    fig, ax = plt.subplot(ncols=1, nrows=1, figsize=(6,6))
    ax.imshow(img)

    for x, y, w, h  in results:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1
        )
        ax.add_patch(rect)
    plt.show()

