#!/usr/bin/python
# coding=utf-8

import cv2 as cv
import time
import numpy as np
from utility.threshold import getTresholdByReExtrem
from img_auto_canny_thresh import auto_canny
import random as rng

mysc = cv.createShapeContextDistanceExtractor()
mync = cv.createHausdorffDistanceExtractor()


def get_main_cnt():
    """
    找到主要的轮廓
    :return:
    """


def match_shape(f1, f2):
    img1 = cv.imread(f1, 0)
    img2 = cv.imread(f2, 0)

    # ########################################################use canny# #####################################
    t1 = time.time()
    canny_thr1 = auto_canny(img1)
    query_canny = cv.Canny(img1, canny_thr1, canny_thr1 * 2, apertureSize=3)

    canny_thr2 = auto_canny(img2)
    test_canny = cv.Canny(img2, canny_thr2, canny_thr2 * 2, apertureSize=3)
    t2 = time.time()
    # ######################################################## use canny# #####################################

    # ##################################### find main contour #####################################
    _, query_contours, _ = cv.findContours(query_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    query_contours.sort(key=lambda ct: cv.arcLength(ct, True))
    # perimeter = cv.arcLength(cnt, True)
    query_mask = np.zeros(query_canny.shape, np.uint8)
    cv.drawContours(query_mask, query_contours, -1, 255, -1)

    _, test_contours, _ = cv.findContours(test_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    test_contours.sort(key=lambda ct: cv.arcLength(ct, True))
    test_mask = np.zeros(test_canny.shape, np.uint8)
    cv.drawContours(test_mask, test_contours, -1, 255, -1)
    t3 = time.time()

    # ##################################### find main contour #####################################
    print('len query_contours={}, len  test_cnt={}, T={}, T2={}'.format(len(query_contours), len(test_contours), t2 - t1, t3 - t2))
    #
    # # t1 = time.time()
    # # ret1 = cv.matchShapes(query_canny, canny_thr2, cv.CONTOURS_MATCH_I1, 0.0)
    # # ret2 = cv.matchShapes(query_canny, canny_thr2, cv.CONTOURS_MATCH_I2, 0.0)
    # # ret3 = cv.matchShapes(query_canny, canny_thr2, cv.CONTOURS_MATCH_I3, 0.0)
    # # t2 = time.time()
    #
    # thr = getTresholdByReExtrem(img1)  # 获取一个合适的阈值
    # ret, query_bin = cv.threshold(img1, thr, 255, cv.THRESH_BINARY)
    # thr2 = getTresholdByReExtrem(img2)  # 获取一个合适的阈值
    # ret, test_bin = cv.threshold(img2, thr2, 255, cv.THRESH_BINARY)
    #
    #
    # # t1 = time.time()
    # # ret1 = cv.matchShapes(query_cnt, test_cnt, cv.CONTOURS_MATCH_I1, 0.0)
    # # ret2 = cv.matchShapes(query_cnt, test_cnt, cv.CONTOURS_MATCH_I2, 0.0)
    # # ret3 = cv.matchShapes(query_cnt, test_cnt, cv.CONTOURS_MATCH_I3, 0.0)
    # # # dis = mysc.computeDistance(query_cnt, test_cnt)
    # # # dis = mync.computeDistance(query_cnt, test_cnt)
    # # t2 = time.time()

    cv.imshow('test_canny', test_canny)
    cv.imshow('test_mask', test_mask)
    cv.waitKey()
    return 0


if __name__ == '__main__':
    f1 = '/disk_workspace/opecv_source/opencv-3.4.6/samples/data/shape_sample/1.png'
    f2 = '/disk_workspace/opecv_source/opencv-3.4.6/samples/data/shape_sample/2.png'

    f3 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_STAND1.jpg'

    f4 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_12_1624_240_1891.jpg'
    f5 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_14_2053_217_2317.jpg'
    f6 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_854_543_1094_803.jpg'
    f7 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_217_1468_513_1763.jpg'
    f8 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_461_4035_688_4244.jpg'
    f9 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_730_2902_993_3146.jpg'

    # ret1 = match_shape(f1, f2)
    ret2 = match_shape(f3, f5)
    ret3 = match_shape(f3, f4)
    ret4 = match_shape(f3, f6)
    ret5 = match_shape(f3, f7)
    ret6 = match_shape(f3, f8)
    ret7 = match_shape(f3, f9)
    # print(ret)
