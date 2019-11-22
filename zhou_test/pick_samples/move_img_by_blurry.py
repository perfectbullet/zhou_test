#!/usr/bin/python
# coding=utf-8

import os
from shutil import copyfile, move
from os.path import join

from get_blurry_res.get_blurry_res import get_blurry_res

import cv2


def move_by_blurry(src_dir, blurry_dir, clear_dir, blurry_thr):
    """
    根据blurry 移动图片
    :return:
    """
    for im_name in os.listdir(src_dir):
        if im_name.endswith('.jpg'):
            src_path = join(src_dir, im_name)
            bl_res = get_blurry_res(cv2.imread(src_path, cv2.IMREAD_GRAYSCALE))
            if bl_res > blurry_thr:
                dst_path = join(clear_dir, im_name)
            else:
                dst_path = join(blurry_dir, im_name)
            move(src_path, dst_path)


def refresh(src_dir, blurry_dir, clear_dir):
    for fn in os.listdir(blurry_dir):
        move(join(blurry_dir, fn), join(src_dir, fn))
    for fn in os.listdir(clear_dir):
        move(join(clear_dir, fn), join(src_dir, fn))
    

if __name__ == '__main__':
    src_dir = '/disk_workspace/test_images/定位环线夹1119/comp_img_for_blurry'
    blurry_dir = join(src_dir, 'blurry')
    clear_dir = join(src_dir, 'clear')
    blurry_thr = 1.5
    move_by_blurry(src_dir, blurry_dir, clear_dir, blurry_thr)
    # refresh(src_dir, blurry_dir, clear_dir)
