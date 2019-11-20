#!/usr/bin/python
# coding=utf-8

from __future__ import print_function
import time
import cv2 as cv
import numpy as np
from scipy.signal import argrelmax, argrelmin, find_peaks_cwt, find_peaks
from matplotlib import pyplot as plt
import random as rng

rng.seed(12345)


# 获取最佳阈值的函数
def getTresholdByGrad(gray, myorder=10):

    hist = np.bincount(gray.ravel(), minlength=256)  # 性能：0.003163 s
    # shape of hist (256, )
    # hist_norm = hist.ravel() / (hist.max() * 0.01)  # 归一化
    # Returns the indices of the maximum values along an axis.
    t1 = time.time()
    max_idx = np.argmax(hist, axis=0)      # 最大值坐标
    t2 = time.time()
    # 计算直方图中的极大值
    min_exts = argrelmin(hist, order=myorder)[0]
    t3 = time.time()
    # 计算直方图中的极小值
    max_exts = argrelmax(hist, order=myorder)[0]
    t4 = time.time()
    print('\n\n\nshape of hist_norm: {}, ndim: {}, max_idx: {}'.format(hist.shape, hist.ndim, max_idx))
    print('\nmax_exts: {} \nmin_exts: {}'.format(max_exts, min_exts))
    print('max_exts perf: {}, min_exts perf: {}, max_idx: {}'.format(t4-t3, t3 - t2, t2 - t1))
    # fig = plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # imgplot = plt.imshow(gray, cmap='gray')
    # # imgplot.set_clim(0.0, 0.7)
    # a.set_title('img')
    # plt.subplot(1, 2, 2)
    # min_ext_X = []
    # min_ext_Y = []
    # for ext in min_exts:
    #     min_ext_X.append(ext)
    #     t = hist[ext]
    #     min_ext_Y.append(t)
    # max_ext_X = []
    # max_ext_Y = []
    # for ext in max_exts:
    #     max_ext_X.append(ext)
    #     t = hist[ext]
    #     max_ext_Y.append(t)
    # plt.plot(min_ext_X, min_ext_Y, 'ro')
    # plt.plot(max_ext_X, max_ext_Y, 'bo')
    # plt.plot(hist)
    # # plt.savefig(file_path.replace('.jpg', '.png'), dpi=300)
    # # plt.close('all')  # 关闭图 0
    # plt.show()

    print('myorder: {}'.format(myorder))
    if myorder < 2:
        mean_thr = np.mean(gray)
        print('mean_thr: {}'.format(mean_thr))
        return np.mean(gray)
    elif min_exts.size < 1 or max_exts.size < 2:
        print('get other_thr for because min_exts: {}, max_exts: {}'.format(min_exts, max_exts))
        myorder -= 1
        other_thr = getTresholdByGrad(gray, myorder=myorder)
        return other_thr
    # 适用于背景为黑色
    if max_idx < 128 and max_idx == max_exts[0]:
        # 最大值和极大值相等
        min_ext_1 = min_exts[0]     # 背景的
        max_ext_0 = max_exts[0]
        max_ext_1 = max_exts[1]
        if max_ext_1 - max_ext_0 > 50:
            print('get other_thr for max scope: {}, myorder: {}'.format(max_ext_1 - max_ext_0, myorder))
            myorder -= 1
            other_thr = getTresholdByGrad(gray, myorder=myorder)
            return other_thr
        elif max_exts[0] < min_ext_1 < max_exts[1]:
            print('return min_ext_1: {}'.format(min_ext_1))
            return min_ext_1
        else:
            myorder -= 1
            other_thr = getTresholdByGrad(gray, myorder=myorder)
            print('get other_thr in A: {}, myorder: {}'.format(other_thr, myorder))
            return other_thr
    elif max_idx < 128 and max_idx != max_exts[0]:
        myorder -= 1
        other_thr = getTresholdByGrad(gray, myorder=myorder)
        print('get other_thr in B: {}, myorder: {}'.format(other_thr, myorder))
        return other_thr
    elif max_idx > 128:
        print('max_idx > 128')
        return np.mean(gray)


def main(img_path):
    src = cv.imread(img_path)
    # Show source image
    cv.imshow('Source Image', src)
    ## [load_image]
    gray = cv.cvtColor(src, code=cv.COLOR_RGB2GRAY)
    thr_b = getTresholdByGrad(gray)
    # thr_b = 40
    print('\n\n\n thr_b: {}'.format(thr_b))
    ## [black_bg]
    # Change the background from white to black, since that will help later to extract
    # better results during the use of Distance Transform
    # src[np.all(src == 255, axis=2)] = 0
    src[np.all(src < thr_b, axis=2)] = 0   # 背景变黑
    # src[np.all(src == 0, axis=2)] = 255
    # Show output image
    # cv.imshow('Black Background Image', src)
    ## [black_bg]

    ## [sharp]
    # Create a kernel that we will use to sharpen our image an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated
    imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    cv.imshow('Laplace Filtered Image', imgLaplacian)
    cv.imshow('New Sharped Image', imgResult)
    ## [sharp]

    ## [bin]
    # Create binary image from source image
    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, thr_b, 255, cv.THRESH_BINARY)
    cv.imshow('Binary Image', bw)
    ## [bin]

    ## [dist]
    # Perform the distance transform algorithm
    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow('Distance Transform Image', dist)
    ## [dist]

    ## [peaks]
    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    cv.imshow('Peaks', dist)
    ## [peaks]

    ## [seeds]
    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')

    # Find total markers
    _, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        # drawContours(image, contours, contourIdx, color, thickness=None,
        # lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # fills the area bounded by the contours if thickness<0
        print('len(contours): {}'.format(len(contours)))
        cv.drawContours(markers, contours, i, (2, ), thickness=-1)

    # Draw the background marker
    # cv.circle(markers, (5,5), 122, (255,255,255), -1)
    cv.imshow('Markers', markers*10000)
    ## [seeds]

    ## [watershed]
    # Perform the watershed algorithm
    cv.watershed(imgResult, markers)

    #mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    # cv.imshow('Markers_v2', imgResult)

    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]

    # Visualize the final image
    cv.imshow('Final Result', dst)
    ## [watershed]

    cv.waitKey()


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
            main(f)
