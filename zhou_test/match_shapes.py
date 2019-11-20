#!/usr/bin/python
# coding=utf-8

import cv2 as cv
import time
import numpy as np


def match_shape(f1, f2):
    img1 = cv.imread(f1, 0)
    img2 = cv.imread(f2, 0)
    ret, thresh = cv.threshold(img1, 127, 255, 0)
    ret, thresh2 = cv.threshold(img2, 127, 255, 0)
    # im2,contours,hierarchy = cv.findContours(thresh,2,1)
    # cnt1 = contours[0]
    # im2,contours,hierarchy = cv.findContours(thresh2,2,1)
    # cnt2 = contours[0]
    t1 = time.time()
    ret1 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I1, 0.0)
    ret2 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I2, 0.0)
    ret3 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I3, 0.0)
    t2 = time.time()
    print('T={}'.format(t2 - t1))
    print(ret1)
    print(ret2)
    print(ret3)
    cv.imshow('f1', thresh)
    cv.imshow('f2', thresh2)
    cv.waitKey()
    return ret


if __name__ == '__main__':
    f1 = '/disk_workspace/test_images/C_TGSE-comp-with-figures/0101_8-3983_1553_1209_shape.png'
    f2 = '/disk_workspace/test_images/C_TGSE-comp-with-figures/0101_8-3984_1591_725_shape.png'

    # f3 = '/disk_workspace/test_images/C_TGSE-comp-with-figures/0101_8-3984_1591_725.jpg'
    # f4 = '/disk_workspace/test_images/C_TGSE-comp-with-figures/0101_8-3984_741_1200.jpg'
    ret = match_shape(f1, f2)
    # print(ret)
