#!/usr/bin/python
# coding=utf-8

from __future__ import print_function
import time
import numpy as np
from scipy.signal import argrelmax, argrelmin
import random as rng

import cv2
import matplotlib.pyplot as plt
from skimage.feature import canny
rng.seed(12345)


def getTresholdByReExtrem(gray, myorder=10):
    """
    get thr by relative extrem-value,
    :param gray: a gray scale img data
    :param myorder: the order means step of thr
    :return: int a better thr
    """
    hist = np.bincount(gray.ravel(), minlength=256)  # performence：0.003163 s
    # Returns the indices of the maximum values along an axis.
    # max_exts perf: 9.60826873779e-05, min_exts perf: 0.000133991241455, max_idx: 1.09672546387e-05
    max_idx = np.argmax(hist, axis=0)
    # calculate relative extrem value
    min_exts = argrelmin(hist, order=myorder)[0]
    max_exts = argrelmax(hist, order=myorder)[0]
    if myorder < 2:
        return np.mean(gray)
    elif min_exts.size < 1 or max_exts.size < 2:
        myorder -= 1
        other_thr = getTresholdByReExtrem(gray, myorder=myorder)
        return other_thr
    # for black background
    if max_idx < 128 and max_idx == max_exts[0]:
        min_ext_1 = min_exts[0]     # 背景的
        max_ext_0 = max_exts[0]
        max_ext_1 = max_exts[1]
        if max_ext_1 - max_ext_0 > 50:
            myorder -= 1
            other_thr = getTresholdByReExtrem(gray, myorder=myorder)
            return other_thr
        elif max_exts[0] < min_ext_1 < max_exts[1]:
            return min_ext_1
        else:
            myorder -= 1
            other_thr = getTresholdByReExtrem(gray, myorder=myorder)
            return other_thr
    elif max_idx < 128 and max_idx != max_exts[0]:
        myorder -= 1
        other_thr = getTresholdByReExtrem(gray, myorder=myorder)
        return other_thr
    else:
        return np.mean(gray)


def segment_TGSE(file_path):
    t1 = time.time()
    gray_img = cv2.imread(file_path, flags=0)
    h, w = gray_img.shape[:2]
    thr = getTresholdByReExtrem(gray_img)
    _, bin_img = cv2.threshold(gray_img, thr, 255, cv2.THRESH_BINARY)
    # remove noisy
    kernel = np.ones((7, 7), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    fig = plt.figure()

    # show gray-img
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    a.set_title('img')

    # show bin_img
    b = fig.add_subplot(2, 2, 2)
    plt.imshow(bin_img, cmap='gray')
    b.set_title('bin_img')

    # *********************Standard Hough Line Transform to find the hough_thr *********************
    # Canny recommends threshold1:threshold2 between 2:1 and 3:1

    # rho, theta 定义了累加平面的分辨率, 他们决定了所获取直线 (theta, rho) 的精确程度,
    rho = 1
    theta = 0.01744  # 1 * math.pi / 180
    # hough_thr 实际指明了直线最少包含的点的绝对数量, 值越大，检测直线精度越高, 可以获得的直线就越少.可能与Hough-votes是等价关系
    # hough_thr = getHoughThreByGrad(canny_img)
    # lines = cv2.HoughLines(canny_img, rho, theta, 100)

    # canny_img1 = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Create the CV_8U version of the distance image It is needed for findContours()
    # Create the marker image for the watershed algorithm
    markers = np.zeros(gray_img.shape, dtype=np.uint8)
    # Find total markers
    _, contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_ls = [cv2.moments(ct)['m00'] for ct in contours]
    area_mean = np.mean(area_ls)
    area_var = np.var(area_ls)
    area_std = np.std(area_ls)
    # print(area_ls)
    # print('area_mean: ', area_mean, 'area_var: ', area_var, 'area_std: ', area_std,)
    contours.sort(key=lambda ct: cv2.moments(ct)['m00'])
    # Draw the foreground markers
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] < 1:
            continue
        # if M['m00'] < area_std:       # 过滤面积小于方差的
        #     continue
        # print('cnt.shape: ', cnt.shape)
        # fill then use cnt to divide
        cv2.drawContours(markers, [cnt, ], 0, (255, ), thickness=-1)
        cv2.drawContours(markers, [cnt, ], 0, (0,), thickness=7)

    d = fig.add_subplot(2, 2, 3)
    plt.imshow(markers, cmap='gray')
    d.set_title('markers')

    # 使用切割后的 markers 再次 寻找轮廓
    markers2 = np.zeros(gray_img.shape, dtype=np.uint8)
    _, contours2, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_ls2 = [cv2.moments(ct)['m00'] for ct in contours2]
    area_std2 = np.std(area_ls2)
    contours2.sort(key=lambda ct: cv2.moments(ct)['m00'])
    # Draw the  markers
    for cnt in contours2:
        M2 = cv2.moments(cnt)
        # if M2['m00'] < area_std2:  # 过滤面积小于方差的区域
        #     continue
        if M2['m00'] < 1:
            continue
        cx = int(M2['m10'] / M2['m00'])
        cy = int(M2['m01'] / M2['m00'])
        # print('cnt2  cx, cy: ', cx, cy)
        # if cx < w * 0.1 or cx > w * 0.9:
        #     continue
        # if cy < h * 0.2 or cy > h * 0.8:
        #     continue
        cv2.drawContours(markers2, [cnt, ], 0, (255,), thickness=-1)

    c = fig.add_subplot(2, 2, 4)
    plt.imshow(markers2, cmap='gray')
    c.set_title('markers2')

    canny_img1 = canny(markers, sigma=3)

    t2 = time.time()
    # 保存markers2
    cv2.imwrite(file_path.replace('.jpg', '_shape.png'), markers2)
    # 保存矩阵
    # np.savetxt(file_path.replace('.jpg', '.txt'), mask2, delimiter=',', fmt='%d')
    plt.savefig(file_path.replace('.jpg', '.png'), dpi=300)     # 慢
    plt.close('all')  # 关闭图 0
    t3 = time.time()
    print('t2 - t1: {}, t3 - t2: {}'.format(t2 - t1, t3 - t2))
    # plt.show()


if __name__ == '__main__':
    from os.path import join
    from os import walk
    rootDir = '/disk_workspace/test_images/C_TGSE-comp-with-figures'
    flag = True
    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:
            if '0101_8-4001_141_258.jpg' not in f and not flag:
                if '.jpg' not in f:
                    continue
                continue
            elif '0101_8-4001_141_258.jpg' in f:
                flag = True

            if '.jpg' not in f:
                continue

            print(f)
            segment_TGSE(f)
