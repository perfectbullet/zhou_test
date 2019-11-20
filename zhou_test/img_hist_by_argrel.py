#!/usr/bin/python
# coding=utf-8

import numpy as np
import os
import time
from os import walk
from os.path import join

import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin, find_peaks_cwt


def auto_canny(image):
    """
    计算一个合适的canny阈值
    :param image: 构件图片
    :return: canny_thr
    """
    i = 1
    std = np.std(image)
    canny_contour_num = []      # canny_im contour  类似于 y = 1 / x  (x > 0)
    while i < 100:      # 0.04s 左右
        canny_im = cv.Canny(image, i, i * 2, apertureSize=3)
        _, contours, _ = cv.findContours(canny_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        canny_contour_num.append(len(contours))
        i += 1
    # 使用轮廓数量 grad_contour_num 类似势能陷阱的图像， 需要求刚好离开陷阱的那个点
    grad_contour_num = np.gradient(canny_contour_num, range(len(canny_contour_num)), edge_order=1)
    min_cnt_exts = argrelmin(grad_contour_num, order=2)[0]

    print('min_cnt_exts: {}'.format(min_cnt_exts))

    min_ext_X = []
    min_ext_Y = []
    for ext in min_cnt_exts:
        min_ext_X.append(ext)
        t = grad_contour_num[ext]
        min_ext_Y.append(t)

    plt.plot(min_ext_X, min_ext_Y, 'ro')
    plt.plot(grad_contour_num)
    plt.show()
    return 10


def hist_argrelextrema(file_path):
    '''
    极值分析
    :param file_path:
    :return:
    '''
    img = cv.imread(file_path, flags=0)
    # gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(img, 5, 75, 75)  # 双边滤波去噪声,加强边缘
    gray = cv.bilateralFilter(gray, 5, 50, 50)  # 双边滤波去噪声,加强边缘
    auto_canny_thr = auto_canny(gray)
    print('auto_canny_thr={}'.format(auto_canny_thr))
    canny_img = cv.Canny(gray, auto_canny_thr, auto_canny_thr*2, apertureSize=3)

    fig = plt.figure()
    c = fig.add_subplot(2, 2, 4)
    plt.imshow(canny_img, cmap='gray')
    c.set_title('canny_img')

    b = fig.add_subplot(2, 2, 3)
    plt.imshow(gray, cmap='gray')
    b.set_title('gray')

    a = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img, cmap='gray')
    # imgplot.set_clim(0.0, 0.7)
    a.set_title('img')

    # hist = cv.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
    hist = np.bincount(gray.ravel(), minlength=256)  # 性能：0.003163 s
    max_hist = hist.max()
    hist_norm = hist.ravel() / (max_hist * 0.01)  # 归一化
    # peakind = find_peaks_cwt(hist_norm, np.arange(0, 256))  # Attempt to find the peaks in a 1-D array.
    min_exts = argrelmin(hist_norm, order=4)[0]
    print('min_exts: {}'.format(min_exts))
    man_exts = argrelmax(hist_norm, order=2)[0]
    plt.subplot(2, 2, 2)
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

    # plt.savefig(file_path.replace('.jpg', '.png'), dpi=300)
    # plt.close('all')  # 关闭图 0
    #
    plt.show()


if __name__ == '__main__':

    rootDir = '/disk_workspace/test_images/支柱角钢1021/角钢构件'

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:

            if '.jpg' not in f or 'C_ZZ_JG' not in f:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f))
            im_path = os.path.join(rootDir, f)
            hist_argrelextrema(im_path)
