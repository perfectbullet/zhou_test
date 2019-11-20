#!/usr/bin/python
# coding=utf-8

import numpy as np
import os
from os import walk
from os.path import join

import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin


def hist_show(file_path):
    img = cv.imread(file_path)
    # com_name, part_name = os.path.basename(file_path)[:-4].split('-')
    # com_idx = com_name.split('_')[-4:]
    # com_idx[2:] = com_idx[:2]
    # part_idx = part_name.split('_')[-4:]
    # part_idx = map(lambda x, y: int(x) - int(y), part_idx, com_idx)
    # part_idx = tuple(part_idx)      # 图片下标

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img, cmap='gray')
    imgplot.set_clim(0.0, 0.7)
    a.set_title('img')

    # 计算直方图
    # hist = cv.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
    # np_hist, np_bins = np.histogram(img.ravel(), 256, [0, 256])
    hist = np.bincount(img.ravel(), minlength=256)  # 性能：0.003163 s
    hist_norm = hist.ravel() / (hist.max() * 0.01)  # 归一化
    # 计算直方图中的极大值

    min_exts = argrelmin(hist_norm, order=8)[0]
    # 计算直方图中的极小值
    man_exts = argrelmax(hist_norm, order=8)[0]
    #
    plt.subplot(1, 2, 2)
    min_ext_X = []
    min_ext_Y = []
    for ext in min_exts:
        min_ext_X.append(ext)
        t = hist_norm[ext]
        min_ext_Y.append(t)
    max_ext_X = []
    max_ext_Y = []
    for ext in man_exts:
        max_ext_X.append(ext)
        t = hist_norm[ext]
        max_ext_Y.append(t)
    plt.plot(min_ext_X, min_ext_Y, 'ro')
    plt.plot(max_ext_X, max_ext_Y, 'bo')
    plt.plot(hist_norm)

    # # 求导
    # X = np.arange(0, 256, 1)
    # grad_list = np.gradient(hist_norm, X, edge_order=1)  # the list elements is always more than zero
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(grad_list)
    #
    # grad2_list = np.gradient(grad_list, X, edge_order=1)  # grad's grad
    # plt.subplot(2, 2, 4)
    # plt.plot(grad2_list)

    # plt.savefig(file_path.replace('.jpg', '.png'), dpi=300)
    # plt.close('all')  # 关闭图 0
    #
    plt.show()


if __name__ == '__main__':

    rootDir = '/disk_workspace/test_images/支柱角钢1021/左正面'

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:

            if '.jpg' not in f:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f))
            im_path = os.path.join(rootDir, f)
            hist_show(im_path)
