#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import caffe
import time, os, sys, cv2

CLASSES = ('__background__', 'person')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

CONF_THRESH = 0.0
NMS_THRESH = 0.3

# timers
_t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
video_suffixes = ['mp4', 'avi', 'm4v']
image_suffixes = ['jpg', 'png', 'JPEG']


def check_folder(path):
    assert os.path.exists(path), 'FOLDER doesn\'t exist: %s' % path


def check_and_make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def output_txt_format(frame_number, rects):
    if len(rects) == 0:
        cur_write_content = str(frame_number) + '\n'
    else:
        rect_array = np.array(rects)
        [m, n] = rect_array.shape
        cur_write_content = str(frame_number) + '\t'
        for cur_rect_num, rect in enumerate(rects):
            for idx, element in enumerate(rect):
                if idx < n - 1:
                    cur_write_content += str(int(round(element))) + ' '
                else:
                    cur_write_content += ('%1.3f' % element)
            if cur_rect_num < m - 1:
                cur_write_content += '\t'
        cur_write_content += '\n'
    return cur_write_content


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def filter_detections(dets, thresh=0.5):
    if len(dets) == 0:
        return []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []
    filtered_dets = dets[inds,:]
    return filtered_dets


def draw_rects(image, class_name, dets, wait_time=0):
    for rect_and_score in dets:
        cur_rect = map(int, rect_and_score[0:4])
        cv2.rectangle(image, (cur_rect[0], cur_rect[1]), (cur_rect[2], cur_rect[3]), (0, 0, 255), 2)
        cv2.putText(image, class_name, (cur_rect[0], cur_rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if wait_time >= 0:
        cv2.imshow("cur show", image)
        cv2.waitKey(wait_time)
    return image


def filter_detect_result(detect_confs, detect_bbox, filter_name):

    for cls_ind, cls in enumerate(CLASSES[0:]):
        cls_ind += 0  # because we skipped background
        cls_boxes = detect_bbox[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = detect_confs[:, cls_ind]
        dets = np.hstack((cls_boxes,
		      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    filtered_dets = filter_detections(dets, CONF_THRESH)
    if len(filtered_dets) == 0:
        return []
    # filtered_dets = filtered_dets[:, 0:4]
    return filtered_dets


def single_video_detect(net, video_path, write_path, filter_name):
    capture = cv2.VideoCapture(video_path)
    write_pid = open(write_path, 'w')
    if not capture.isOpened():
        return
    frame_count = 0
    while True:
        state, cur_frame = capture.read()
        frame_count += 1
        print frame_count
        if not state:
            break
        scores, boxes = im_detect(net, cur_frame, _t, None)

        detect_bboxs = filter_detect_result(scores, boxes, filter_name)
        draw_rects(cur_frame, 'person', detect_bboxs, wait_time=0)
        cur_write_content = output_txt_format(frame_count, detect_bboxs)
        write_pid.write(cur_write_content)
        # draw_rects(cur_frame, filter_name, detect_bboxs, 1)
    write_pid.close()


def video_detect(data_dir, out_dir, detect_net, filter_name='person'):
    check_folder(data_dir)
    video_names = os.listdir(data_dir)
    for cur_video_name in video_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'cur processed video name:{}'.format(cur_video_name)
        cur_video_path = os.path.join(data_dir, cur_video_name)
        cur_write_path = os.path.join(out_dir, cur_video_name.split('.')[0] + '.txt')
        single_video_detect(detect_net, cur_video_path, cur_write_path, filter_name)


def single_image_folder_detect(data_dir, out_dir, detect_net, filter_name='__background__'):
    check_folder(data_dir)
    check_and_make_folder(out_dir)

    txt_write_path = os.path.join(out_dir, os.path.basename(data_dir) + '.txt')
    write_pid = open(txt_write_path, 'w')
    img_list = os.listdir(data_dir)

    img_count = 0
    for cur_img_name in img_list:
        img_count += 1
        cur_img_path = os.path.join(data_dir, cur_img_name)
        cur_img_data = cv2.imread(cur_img_path)
        if cur_img_data is None:
            print 'Warning ,image read error:%s' % cur_img_path
            continue
        scores, boxes = im_detect(detect_net, cur_img_data, _t, None)
        print scores
        print boxes
        detect_result = filter_detect_result(scores, boxes, filter_name)
        draw_rects(cur_img_data, filter_name, detect_result, wait_time=0)
        cur_write_content = output_txt_format(cur_img_name.split('.')[0], detect_result)
        write_pid.write(cur_write_content)
        print '%d, path:%s' % (img_count, cur_img_path)
    write_pid.close()


def image_detect(data_dir, out_dir, detect_net, filter_name='person'):
    check_folder(data_dir)
    img_folder_names = os.listdir(data_dir)
    for cur_image_name in img_folder_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'cur processed folder name:'.format(cur_image_name)
        cur_dir_path = os.path.join(data_dir, cur_image_name)
        single_image_folder_detect(cur_dir_path, out_dir, detect_net, filter_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('-i', dest='input_data_dir', help='input data dir')
    parser.add_argument('-o', dest='output_data_dir', help='output data dir')
    parser.add_argument('-f', dest='filter_name', help='filter classes', default='person')

    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg_from_file('../experiments/cfgs/pva_lite_end2end.yml')

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = '../models/pvanet/lite/test.pt'
    caffemodel = '../output/pva_lite_end2end/renren1_trainval/pva_lite_faster_rcnn_iter_110000.caffemodel'

    data_dir = args.input_data_dir
    out_dir = args.output_data_dir
    filter_name = args.filter_name
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # case single video input
    single_image_folder_detect(data_dir, out_dir, net, '__background__')

'''
    if os.path.isfile(data_dir) and data_dir.split('.')[1] in video_suffixes:
        single_video_detect(net, data_dir, out_dir, 'person')
    elif os.path.isdir(data_dir):
        sub_files = os.listdir(data_dir)
        file_split = sub_files[0].split('.')

        # case multi image folder
        if len(file_split) == 1:
            image_detect(data_dir, out_dir, net)
        # case multi video files
        elif file_split[1] in video_suffixes:
            video_detect(data_dir, out_dir, net, 'person')
        # case one image folder
        elif file_split[1] in image_suffixes:
            single_image_folder_detect(data_dir, out_dir, net, 'person')
        else:
            print 'please input legal input data dir!!!'
'''




