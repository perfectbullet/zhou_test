#!/usr/bin/python
# coding=utf-8

import numpy as np
import os
from os import walk
from os.path import join

import cv2 as cv
from matplotlib import pyplot as plt


def grap_cut(file_path):
    img = cv.imread(file_path)
    com_name, part_name = os.path.basename(file_path)[:-4].split('-')
    com_idx = com_name.split('_')[-4:]
    com_idx[2:] = com_idx[:2]
    part_idx = part_name.split('_')[-4:]
    part_idx = map(lambda x, y: int(x) - int(y), part_idx, com_idx)
    part_idx = tuple(part_idx)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = part_idx
    print('img2.shape: {}, markers.shape: {}'.format(img.shape, mask.shape))
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


if __name__ == '__main__':

    rootDir = '/disk_workspace/test_images/支柱角钢0923/pickledata'

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:

            if '.jpg' not in f:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f))
            im_path = os.path.join(rootDir, f)
            grap_cut(im_path)
