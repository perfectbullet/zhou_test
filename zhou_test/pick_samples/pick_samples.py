#!/usr/bin/python
# coding=utf-8

import sys
import os
import time
import pickle
from os import walk
from os.path import join

import cv2


def loadPicklePath(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_to(impath, img):
    """
    保存图片
    :param impath:
    :param img:
    :return: None
    """
    su = cv2.imwrite(impath, img)
    if not su:
        print('failed fname={}'.format(impath))
        sys.exit(1)


def main():
    base_dir = '/disk_workspace/test_images/达州至凉雾-双线201806鱼背山-罗田区间/'
    rootDir = join(base_dir, 'pickledata')
    comp_img_sclt06 = join(base_dir, 'comp_img_sclt06')  # 分数小于0.6
    comp_img_scgt06 = join(base_dir, 'comp_img_scgt06')  # 分数大于0.6
    comp_img_scgt08 = join(base_dir, 'comp_img_scgt08')  # 分数大于0.8
    comp_img_scgt09 = join(base_dir, 'comp_img_scgt09')  # 分数大于0.9

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        print('root={}, len files={}'.format(root, len(files)))
        for f in files:
            if '.txt' not in f:
                continue
            # print('fname={}'.format(f))
            task = loadPicklePath(os.path.join(rootDir, f))
            task.xml.removeDupAreas(0.2)
            careas = task.xml.getAreasByFlag("C_DWHXJ")  # 获取名称以C_开头的，cArea区域
            if not careas:
                continue
            task.load()
            for ca in careas:
                ca_img = task.img.getImgROI(ca)
                ca_score = ca.score
                fname = 'sc_{:.3f}_{}_{}'.format(ca_score, str(ca), '.jpg')
                if ca_score <= 0.6:
                    write_to(os.path.join(comp_img_sclt06, fname), ca_img)
                if ca_score > 0.6:
                    write_to(os.path.join(comp_img_scgt06, fname), ca_img)
                if ca_score > 0.8:
                    write_to(os.path.join(comp_img_scgt08, fname), ca_img)
                if ca_score > 0.9:
                    write_to(os.path.join(comp_img_scgt09, fname), ca_img)


if __name__ == '__main__':
    main()
