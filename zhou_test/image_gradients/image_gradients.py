#!/usr/bin/python
# coding=utf-8

import os
import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def show(img_path, src_img, laplacian, abs_lap64f, scharr_img):
    lap_txt = 'laplacian shape={} type={} std={:.2f}\n mean={:.2f} restd={} min={}'\
            .format(laplacian.shape,  laplacian.dtype,  laplacian.std(), np.mean(np.absolute(laplacian)), laplacian.std()/np.mean(np.absolute(laplacian)), laplacian.min())
    lap64_txt = 'abs_lap64f shape={} type={} std={:.2f}\n mean={:.2f} max={} min={}'\
        .format(abs_lap64f.shape,  abs_lap64f.dtype,  abs_lap64f.std(), np.median(abs_lap64f), abs_lap64f.max(), abs_lap64f.min())
    scharr_txt = 'scharr shape={} type={} std={:.2f}\n mean={:.2f} restd={} min={}'\
        .format(scharr_img.shape,  scharr_img.dtype,  scharr_img.std(), np.mean(np.absolute(scharr_img)), laplacian.std()/np.mean(np.absolute(laplacian)), scharr_img.min())
    # print(scharr_txt)
    plt.subplot(2, 2, 1), plt.imshow(src_img, cmap='gray')
    plt.title('Original', fontsize=6),  plt.xticks([]),  plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title(lap_txt, fontsize=6),  plt.xticks([]),  plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(abs_lap64f, cmap='gray')
    plt.title(lap64_txt, fontsize=6),  plt.xticks([]),  plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(scharr_img, cmap='gray')
    plt.title(scharr_txt, fontsize=6), plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.savefig(img_path.replace('.jpg', '.png'), dpi=500)
    # plt.close()


def main(img_path):
    """
    预处理图片, 然后展示
    T = 0.05
    :param img_path:
    :param disnoising:
    :return: blurry_res     模糊程度
    """
    gray_img = cv.imread(img_path, 0)
    t1 = time.time()
    scharr_dx = cv.Scharr(gray_img, cv.CV_64F, 1, 0)
    scharr_dy = cv.Scharr(gray_img, cv.CV_64F, 0, 1)
    scharr_img = cv.addWeighted(scharr_dx, 0.5, scharr_dy, 0.5, 0)
    laplacian = cv.Laplacian(gray_img, cv.CV_64F)
    abs_lap64f = np.absolute(laplacian)
    scharr_std = scharr_img.std()
    lap_std = laplacian.std()
    abs_lap_std = abs_lap64f.std()

    disnoy_img = cv.fastNlMeansDenoising(gray_img, h=3, templateWindowSize=7, searchWindowSize=21)  # T=0.1 ~ 0.01
    di_scharr_dx = cv.Scharr(disnoy_img, cv.CV_64F, 1, 0)
    di_scharr_dy = cv.Scharr(disnoy_img, cv.CV_64F, 0, 1)
    di_scharr_img = cv.addWeighted(di_scharr_dx, 0.5, di_scharr_dy, 0.5, 0)
    di_laplacian = cv.Laplacian(disnoy_img, cv.CV_64F)
    di_abs_lap64f = np.absolute(di_laplacian)
    di_scharr_std = di_scharr_img.std()
    di_lap_std = di_laplacian.std()
    di_abs_lap_std = di_abs_lap64f.std()
    
    beta = max(abs_lap_std, di_abs_lap_std) * 0.8 + max(lap_std, di_lap_std) * 0.5 + max(scharr_std, di_scharr_std) * 0.04
    beta /= 3.0

    t2 = time.time()
    print('T={}, beta={}, img_path={}'.format(t2 - t1, beta, os.path.basename(img_path)))
    show(img_path, gray_img, laplacian, abs_lap64f, scharr_img)


def main_dir(img_dir):
    fns = [fn for fn in os.listdir(img_dir) if fn.endswith('.jpg')]
    for fn in fns:
        main(os.path.join(img_dir, fn))
    

if __name__ == '__main__':
    """
    laplacian shape=(357, 313),  type=float64,  std=10.7155648791, mean=0.00136028852436, max=182.0, min=-159.0
    abs_lap64f shape=(357, 313),  type=float64,  std=9.46642548953, mean=5.02096813166, max=182.0, min=0.0
    """
    clear_img_path = 'clear.jpg'
    '''
    laplacian shape=(238, 212),  type=float64,  std=2.0830761685, mean=0.00033692722372, max=36.0, min=-99.0
    abs_lap64f shape=(238, 212),  type=float64,  std=1.93851787909, mean=0.762466307278, max=99.0, min=0.0
    '''
    blurry_img_path = '/disk_workspace/test_images/定位环线夹1119/comp_img_scgt06/sc_1.000_C_DWHXJ_2590_2480_2910_2837_.jpg'
    
    main(blurry_img_path)
    # main(blurry_img_path)
    
    # img_dir = '/disk_workspace/train_data_for_svm/dwhx_samples_20191120/comp_img_scgt08'
    # main_dir(img_dir)
