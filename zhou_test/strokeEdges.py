#!/usr/bin/python
# coding=utf-8

import time

import cv2


def strokeEdges(src, blurKsize=5, edgeKsize=3):
    graySrc = src.copy()
    if blurKsize >= 3:
        graySrc = cv2.medianBlur(graySrc, blurKsize)
    graySrc = cv2.Laplacian(graySrc, cv2.CV_8U, ksize=edgeKsize)
    # 归一化
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channel = graySrc * normalizedInverseAlpha
    # channels = cv2.split(src)
    # for channel in channels:
    #     channel[:] = channel * normalizedInverseAlpha
    return channel
    # return cv2.merge([channel, ])


def main(img_path):
    src = cv2.imread(img_path, 0)
    t1 = time.time()
    dst = strokeEdges(src)
    t2 = time.time()
    print('T={}'.format(t2 - t1))
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('C_ZZ_JG_1079_1749_1210_1876.jpg')
