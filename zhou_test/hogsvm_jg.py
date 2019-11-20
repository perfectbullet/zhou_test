#!/usr/bin/python
# coding=utf-8

import cv2 as cv
import numpy as np
import time


class HOG_Detector(object):
    """
    计算图像的 hog descriptor
    """
    def __init__(self, win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9, svm_data_path='svm_data_jg_20191112.dat'):
        self.win_size = win_size  # 窗口大小, 就是图像的shape
        self.block_size = block_size  # block，
        self.block_stride = block_stride  # block step
        self.cell_size = cell_size  # cell size, 梯度直方图采样框
        self.nbins = nbins  # hist bins, 和梯度直方图的粒度有关
        # HOG 描述符计算器
        self.hog_det = cv.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
        self.svmer = cv.ml.SVM_load(svm_data_path)      # 向量机
        self.lable2code = {1: 'C_ZZ_JG_ZM', -1: 'C_ZZ_JG_BM'}

    def _get_hog_data(self, gray_img):
        """
        计算图像 hog descriptor, 先缩小， 在计算 hog descriptor
        :param gray_img:    构件灰度图
        :return: ndarray
        """
        t1 = time.time()
        dst = cv.resize(gray_img, self.win_size, interpolation=cv.INTER_AREA)
        t2 = time.time()
        print('_reshape_image T={}'.format(t2 - t1))
        return self.hog_det.compute(dst, winStride=self.block_stride, padding=(0, 0))

    def get_result(self, gary_img):
        """
        获取一个分类的结果
        :param gary_img: 构件灰度图
        :return: 一个分类的结果 （int->code, str->name）
        """
        query_data = self._get_hog_data(gary_img)
        query_data_pi = np.array([query_data, ])    #
        qurey_re = self.svmer.predict(query_data_pi)[1]
        return int(qurey_re), self.lable2code.get(int(qurey_re))


if __name__ == '__main__':
    """
    使用 SVM 对构件进行二分类：
    训练数据：
        1. 支柱角钢分正面和背面， 
    """
    from os import walk
    from os.path import join

    # img_dir = '/disk_workspace/test_images/支柱角钢1021/zzjg/'

    # for root, dirs, files in walk(img_dir):
    #     files = [join(root, name) for name in files]
    #     for f in files:
    #         print(f)

    # jj_zm = '/disk_workspace/test_images/支柱角钢1021/zzjg/zzjg_zm'
    # jj_bm = '/disk_workspace/test_images/支柱角钢1021/zzjg/zzjg_bm'
    # train_data, labels = get_dataset(jj_zm, jj_bm)
    # # train_data.shape = (len(train_data), 1764), labels.shape = (len(train_data),)
    # print('train_data.shape={},  labels.shape={}'.format(train_data.shape, labels.shape))
    # svm = cv.ml.SVM_create()
    # svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setType(cv.ml.SVM_C_SVC)
    # svm.setC(2.67)
    # svm.setGamma(5.383)
    # svm.train(train_data, cv.ml.ROW_SAMPLE, np.array(labels))
    # svm.save('svm_data_jg_20191112.dat')

    hog_dtector = HOG_Detector()
    test_img = '/disk_workspace/test_images/支柱角钢1021/zzjg/zzjg_zm/C_ZZ_JG_4778_808_5071_1083.jpg'
    query_img = cv.imread(test_img, 0)
    t1 = time.time()
    result = hog_dtector.get_result(query_img)
    t2 = time.time()
    print('result={}, T={}'.format(result, t2 - t1))


