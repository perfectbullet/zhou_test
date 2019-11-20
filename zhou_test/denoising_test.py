#!/usr/bin/python
# coding=utf-8

import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from scipy.interpolate import spline


if __name__ == '__main__':
    img = cv.imread('C_ZZ_JG_1079_1749_1210_1876.jpg')
    t1 = time.time()
    dst = cv.fastNlMeansDenoising(img, h=5, templateWindowSize=7, searchWindowSize=21)
    t2 = time.time()
    print('T={}'.format(t2 - t1))

    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    plt.show()


