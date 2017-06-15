import pickle
import numpy as np
import selectivesearch # pip install selectivesearch

from skimage import io as skimage_io
from PIL import Image


def pil_to_nparray(pil_image):
    """
    using PIL library transform image to float32
    :param pil_image:
    :return:
    """
    pil_image.load()
    return np.asarray(pil_image, dtype='float32')


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=Image.ANTIALIAS):
    """
    resize image with resize_mode to new dimension
    :param in_image:
    :param new_width:
    :param new_height:
    :param out_image:
    :param resize_mode:
    :return:
    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a,
                    xmin_b, xmax_b, ymin_b, ymax_b):
    """
    if intersection:judge a in b or not.
    :param xmin_a: rectangle a minimize x
    :param xmax_a: rectangle a maximize x
    :param ymin_a: rectangle a minimize y
    :param ymax_a: rectangle a maximize y
    :param xmin_b: rectangle b minimize x
    :param xmax_b: rectangle b maximize x
    :param ymin_b: rectangle b minimize y
    :param ymax_b: rectangle b maximize y
    :return: if rectangle a and b have common area then return the interaction area's square
             else return false
    """
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


def IoU(ver1, vertice2):
    """
    calculate IoU of two area
    :param ver1: (original x, original y, width, height)
    :param vertice2:(left_x,top_y,right_x,bottom_y, width, height)
    :return:
    """
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[1], vertice1[2], vertice1[3],
                                 vertice2[0], vertice2[1], vertice2[2], vertice2[3])
    if area_inter:
        area_1 = ver1[2]*ver1[3]
        area_2 = vertice2[4]*vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    else:
        return False


def clip_pic(img, rect):
    """
    clip image with rectangle
    :param img:
    :param rect:
    :return:
    """
    x_left = rect[0]
    y_top = rect[1]
    w = rect[2]
    h = rect[3]
    x_right = x_left+w
    y_bottom = y_top+h
    return img[x_left:x_right, y_top:y_bottom], [x_left, y_top, x_right, y_bottom, w, h]


def load_train_proposals(datafile, num_class, threshold=0.5, svm=False, save=False, save_path='dataset.pkl'):
    """
    loading training dataset and using selective search method to generate proposals
    :param datafile:
    :param num_class:
    :param threshold:
    :param svm:
    :param save:
    :param save_path:
    :return:
    """
    train_list = open(datafile, 'r')
    labels = []
    images = []
    for line in train_list:
        #tmp[0] = image_address,tmp[1]=label,tmp[2]=rectangle vertices
        tmp = line.strip().split(' ')
        img = skimage_io.imread(tmp[0])
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

        candidates = set()

        print('... processing image %s' % tmp[0])

        # iteration every region
        for r in regions:
            # excluding same rectangle
            if r['rect'] in candidates:
                continue
            if r['size'] < 220:
                continue
            # resize to 224*224 for input
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # delete empty array
            if len(proposal_img) == 0:
                continue
            # check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a==0 or b==0 or c==0:
                continue
            im = Image.fromarray(proposal_img)
            resized_proposal_img = resize_image(im, 224, 224)
            candidates.add(r['rect'])
            img_float = pil_to_nparray(resized_proposal_img)
            images.append(img_float)
            # IoU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IoU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if svm==False:
                label = np.zeros(num_class+1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
            else:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
        # end of for
    # end of for
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels


def load_from_pkl(dataset_file):
    """
    pickle loading dataset
    :param dataset_file:
    :return:
    """
    X, Y= pickle.load(open(dataset_file, 'rb'))
    return X, Y