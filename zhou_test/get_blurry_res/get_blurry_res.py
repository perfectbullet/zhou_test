#!/usr/bin/python
# coding=utf-8

import time

import numpy as np
import cv2


def get_blurry_res(gray_img):
    """
    by zj
    获取图片的模糊程度, 目前只在定位环线夹上测试过
    还有调整的空间
    T = 0.05
    :param img_path:
    :param disnoising:
    :return: blurry_res     模糊程度,  1 作为分界线, 越比 1 小越模糊, 越比 1 大越清晰
    """
    t1 = time.time()
    scharr_dx = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0)
    scharr_dy = cv2.Scharr(gray_img, cv2.CV_64F, 0, 1)
    scharr_img = cv2.addWeighted(scharr_dx, 0.5, scharr_dy, 0.5, 0)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    abs_lap64f = np.absolute(laplacian)
    scharr_std = scharr_img.std()
    lap_std = laplacian.std()
    abs_lap_std = abs_lap64f.std()
    # disnoising
    disnoy_img = cv2.fastNlMeansDenoising(gray_img, h=3, templateWindowSize=7, searchWindowSize=21)  # T=0.1 ~ 0.01
    di_scharr_dx = cv2.Scharr(disnoy_img, cv2.CV_64F, 1, 0)
    di_scharr_dy = cv2.Scharr(disnoy_img, cv2.CV_64F, 0, 1)
    di_scharr_img = cv2.addWeighted(di_scharr_dx, 0.5, di_scharr_dy, 0.5, 0)
    di_laplacian = cv2.Laplacian(disnoy_img, cv2.CV_64F)
    di_abs_lap64f = np.absolute(di_laplacian)
    di_scharr_std = di_scharr_img.std()
    di_lap_std = di_laplacian.std()
    di_abs_lap_std = di_abs_lap64f.std()
    beta = max(abs_lap_std, di_abs_lap_std) * 0.8 + max(lap_std, di_lap_std) * 0.5 + max(scharr_std, di_scharr_std) * 0.04
    beta = beta / 3.0
    t2 = time.time()
    print('get_blurry_res T={:.4f}'.format(t2 - t1))
    return beta
