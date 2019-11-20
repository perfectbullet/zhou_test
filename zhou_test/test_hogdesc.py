#!/usr/bin/python
# coding=utf-8

import cv2 as cv
import numpy as np
import os

# 把目标图放在64x128的灰色图片中间，方便计算描述子
def get_hog_descriptor(image):
    hog = cv.HOGDescriptor()
    h, w = image.shape[:2]
    rate = 64 / float(w)
    image = cv.resize(image, (64, np.int(rate*h)))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bg = np.zeros((128, 64), dtype=np.uint8)
    bg[:,:] = 127
    h, w = gray.shape
    dy = (128 - h) // 2
    bg[dy:h+dy,:] = gray
    descriptors = hog.compute(bg, winStride=(8, 8), padding=(0, 0))
    return descriptors

def get_data(train_data, labels, path, lableType):
    for file_name in os.listdir(path):
        img_dir = os.path.join(path, file_name)
        img = cv.imread(img_dir)
        hog_desc = get_hog_descriptor(img)
        one_fv = np.zeros([len(hog_desc)], dtype=np.float32)
        for i in range(len(hog_desc)):
            one_fv[i] = hog_desc[i][0]
        train_data.append(one_fv)
        labels.append(lableType)
    return train_data, labels

def get_dataset(pdir, ndir):
    train_data = []
    labels = []
    # 获取正样本
    train_data, labels =  get_data(train_data, labels, pdir, lableType=1)
    # 获取负样本
    train_data, labels =  get_data(train_data, labels, ndir, lableType=-1)

    return np.array(train_data, dtype=np.float32), np.array(labels, dtype=np.int32)


if __name__ == '__main__':
    # train_data的shape为(n, 3780), labels(n,)
    # n为样本数
    train_data, labels = get_dataset("/disk_workspace/test_images/支柱角钢1021/zzjg/zzjg_zm/", "/disk_workspace/test_images/支柱角钢1021/zzjg/zzjg_bm/")