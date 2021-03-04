import sys
import os
import time
import logging
import numpy as np
import argparse
import json, re
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.common import read_imgfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([body_part.x * image_w, body_part.y * image_h, body_part.score])
    return keypoints


def base_test():
    model = 'mobilenet_thin'
    resize = '432x368'
    w, h = model_wh(resize)

    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    image_path = "JZ20-R-TRU5400-012-1-03.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    image = common.read_imgfile(image_path, None, None)

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    max_prob = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(max_prob)
    plt.show();

    plt.figure(figsize=(15, 8))
    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
    plt.imshow(bgimg, alpha=0.5)
    plt.imshow(max_prob, alpha=0.5)
    plt.colorbar()
    plt.show()

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    plt.imshow(image)
    plt.show()


def save_to_file():
    model = 'mobilenet_thin'
    resize = '432x368'
    w, h = model_wh(resize)

    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    image_path = "JZ20-R-TRU5400-012-1-03.jpg"
    image = common.read_imgfile(image_path, None, None)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    annotation = dict()
    annotation["version"] = 1.3
    annotation["people"] = []
    for human in humans:
        res = write_coco_json(human, image.shape[0], image.shape[1])
        human_annotation = dict()
        human_annotation['person_id'] = [-1]
        human_annotation['pose_keypoints_2d'] = res
        human_annotation['face_keypoints_2d'] = []
        human_annotation['hand_left_keypoints_2d'] = []
        human_annotation['hand_right_keypoints_2d'] = []
        human_annotation['pose_keypoints_3d'] = []
        human_annotation['face_keypoints_3d'] = []
        human_annotation['hand_left_keypoints_3d'] = []
        human_annotation['hand_right_keypoints_3d'] = []
        annotation["people"].append(human_annotation)

    with open('{}.json'.format(image_path.split('.')[0]), 'w') as fp:
        json.dump(annotation, fp)


if __name__ == '__main__':
    save_to_file()
