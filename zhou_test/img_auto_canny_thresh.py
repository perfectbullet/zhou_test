#!/usr/bin/python
# coding=utf-8

import numpy as np
import os
import time
from os import walk
from os.path import join

import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin, find_peaks, peak_prominences, peak_widths
# from skimage.exposure import adjust_gamma
from utility.threshold import getTresholdByReExtrem
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy import ndimage


# cv.createShapeContextDistanceExtractor
iter_tims = 100
_super_value = 0.001
_aperture_size = 3

STAND_JG1 = '/disk_workspace/test_images/支柱角钢1021/左正面/C_ZZ_JG_STAND1.jpg'
_IMA1 = cv.imread(STAND_JG1, flags=0)
_IMA1 = cv.bilateralFilter(_IMA1, 5, 75, 75)  # 双边滤波去噪声,加强边缘  T=0.07s
_IMA1 = cv.bilateralFilter(_IMA1, 5, 50, 50)  # 双边滤波去噪声,加强边缘  T=0.07s
thr = getTresholdByReExtrem(_IMA1)  # 获取一个合适的阈值
_, _THRE_IMA1 = cv.threshold(_IMA1, thr, 255, cv.THRESH_BINARY)


def match_shape(thresh, thresh2):
    t1 = time.time()
    ret1 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I1, 0.0)
    ret2 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I2, 0.0)
    ret3 = cv.matchShapes(thresh, thresh2, cv.CONTOURS_MATCH_I3, 0.0)
    t2 = time.time()
    print('T={}, ret1={}, ret2={}, ret3={}'.format(t2 - t1, ret1, ret2, ret3))


def smooth(arr, step=2):
    start = step - step / 2
    end = 256 - step / 2
    res = np.zeros((256, ))
    for i in range(start, end):
        temp = 0.0
        for j in range(0-step / 2, step / 2):
            temp += arr[i + j]
        temp /= step
        res[i] = int(temp)
    return res


def auto_canny(image, ):
    """
    by zj
    计算一个合适的canny阈值 thr1 第二个阈值一般是 thr1 的 2 到三倍
    性能 0.05s 左右(C_ZZ_JG 和 套管双耳实测是0.05s左右， 图片大小一般是 250 × 250)
    图片过大的会增加计算时间计算时间和图像的size（w × h）正相关
    :param image: 构件图片
    :return: canny_thr
    """
    t1 = time.time()
    i = 1
    canny_contour_num = []      # canny_im contour  类似于 y = 1 / x  (x > 0) 的一个分布
    while i <= iter_tims:      # 0.04s
        canny_im = cv.Canny(image, i, i * 2, apertureSize=_aperture_size)
        _, contours, _ = cv.findContours(canny_im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        canny_contour_num.append(len(contours))
        i += 1
    nums_temp = np.array(canny_contour_num)
    # normalize
    cnt_nums_norm = nums_temp / np.linalg.norm(nums_temp)   # shape = (100, )
    # 使用轮廓数量 grad_contour_num 类似势能陷阱的图像， 需要求刚好离开陷阱的那个点
    grad_contour_num = np.gradient(cnt_nums_norm, range(len(cnt_nums_norm)), edge_order=1)
    std_ls = [grad_contour_num[idx:].std() for idx in range(iter_tims)]
    std_ls = np.array(std_ls)
    idx_array = np.where(std_ls < _super_value)[0]
    rest = idx_array[0]
    t2 = time.time()
    # print('cnt_nums_norm  shape={}, mean={:.4f}, std={:.4f}, contour len={}, T={:.5f}'.format(cnt_nums_norm.shape,
    #     cnt_nums_norm.mean(), cnt_nums_norm.std(), canny_contour_num[rest], t2 - t1))
    return rest


def find_real_peaks(hist_norm, min_relhight=0.01):
    """
    定位波峰   T=0.0004
    :param hist_norm: ndarray 归一化的直方图
    :return: real_peaks_idx ndarray 波峰在直方图中的横坐标
    """
    rel_max, = argrelmax(hist_norm, order=3)    # 极大值横坐标
    # Calculate the prominence of each peak in a signal.

    # 如果 prominences 的值(这里指的是元素的值) 小于了 0.001就不算一个peak
    widths, width_heights, left_ips, right_ips = peak_widths(hist_norm, rel_max, rel_height=0.618)
    # width_heights is the height of the contour lines at which the widths where evaluated.
    # calculate contour_width_heights by width_heights and peak heights
    # 如果 contour_width_heights 的值(这里指的是元素的值) 小于了 0.001就不算一个peak, 因为这个值是"半"峰高
    contour_width_heights = hist_norm[rel_max] - width_heights
    # all_areas = widths * width_heights  # 这个作为计算峰面积不合适
    all_areas = widths * contour_width_heights  # 半峰高和
    max_hist = hist_norm.max()
    print('max_hist={}, 0.01 * max_hist={}, 0.1 * max_hist={}'.format(max_hist, 0.01 * max_hist, 0.1 * max_hist))
    temp_idx1 = contour_width_heights > 0.01 * max_hist
    temp_idx2 = all_areas > 0.1 * max_hist
    temp_idx = np.where(temp_idx1 & temp_idx2)
    real_peaks_idx = rel_max[temp_idx]
    # print('real_peaks_idx={}'.format(real_peaks_idx))
    print('width_height={}'.format(width_heights[temp_idx]))
    print('contour_width_heights={}'.format(contour_width_heights[temp_idx]))
    print('widths={}'.format(widths[temp_idx]))
    print('areas={}'.format(all_areas[temp_idx]))
    print('real_peaks_idx={}'.format(real_peaks_idx))
    return real_peaks_idx


def hist_argrelextrema(file_path):
    """
    极值分析
    :param file_path:
    :return:
    """
    img = cv.imread(file_path, flags=0)
    # gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    img = np.where(img == 0, 1, img)
    t1 = time.time()
    gray = cv.bilateralFilter(img, 5, 75, 75)  # 双边滤波去噪声,加强边缘  T=0.07s
    gray = cv.bilateralFilter(gray, 5, 50, 50)  # 双边滤波去噪声,加强边缘  T=0.07s
    t2 = time.time()
    # enhance
    # mean = np.mean(gray)
    # std = np.std(gray)
    # print('mean={:.5f}, std={:.5f}'.format(mean, std))
    # if mean < 30.0 and std < 15:
    #     gamma_table = np.round(np.power(np.arange(256) / 255., 0.5) * 255.).astype(np.uint8)  # 构建gamma变换表
    #     gray = cv.LUT(gray, gamma_table)
    #     # gray = adjust_gamma(gray, 0.5)
    # enhance
    auto_canny_thr = auto_canny(gray)
    mean = np.mean(gray)
    std = np.std(gray)
    canny_img = cv.Canny(gray, auto_canny_thr, auto_canny_thr*2, apertureSize=_aperture_size)
    _, contours, _ = cv.findContours(canny_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print('mean={:.5f}, std={:.5f} auto_canny_thr={}, len(contors)={}, T={:.5f}'
          .format(mean, std, auto_canny_thr, len(contours), t2 - t1))

    fig = plt.figure()
    c = fig.add_subplot(2, 3, 1)
    plt.imshow(canny_img, cmap='gray')
    c.set_title('canny_img')

    b = fig.add_subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    b.set_title('gray')

    a = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(img, cmap='gray')
    # imgplot.set_clim(0.0, 0.7)
    a.set_title('img')

    # 计算直方图
    hist = cv.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
    # hist = np.bincount(gray.ravel(), minlength=256)  # 性能：0.003163 s
    # hist_norm = hist.copy()
    # hist_norm1 = hist / np.linalg.norm(hist)     # normalize menthod or x' = (x - μ)／σ
    hist_norm = cv.normalize(hist, None, norm_type=cv.NORM_L2)  # normalize menthod or x' = (x - μ)／σ
    hist_norm = hist_norm.reshape(hist_norm.shape[0], )
    # smooth
    smooth_gray = cv.boxFilter()
    sm_hist = cv.calcHist([gray], [0], None, [256], [0, 256])  # 性能：0.025288 s
    sm_hist_norm = cv.normalize(sm_hist, None, norm_type=cv.NORM_L1)  # normalize menthod or x' = (x - μ)／σ
    sm_hist_norm = hist_norm.reshape(sm_hist_norm.shape[0], )
    print('sm_hist_norm.shape={}  hist_norm.shape={}'.format(sm_hist_norm.shape, hist_norm.shape))
    # hist_sum0 = cv.normalize(gray.sum(axis=0), None, alpha=1, beta=0, norm_type=cv.NORM_L1)
    # hist_sum1 = cv.normalize(gray.sum(axis=1), None, alpha=1, beta=0, norm_type=cv.NORM_L1)
    norm_gray = gray - gray.mean() / gray.std()
    thr = getTresholdByReExtrem(gray)   # 获取一个合适的阈值
    _, thre_img = cv.threshold(gray, thr, 255, cv.THRESH_BINARY)
    match_shape(_THRE_IMA1, thre_img)

    b = fig.add_subplot(2, 3, 6)
    plt.imshow(thre_img, cmap='gray')
    b.set_title('thre_img')

    t3 = time.time()
    sum0 = thre_img.sum(axis=0)
    # norm_sum0 = sum0 - sum0.mean() / sum0.std()
    subplt5 = plt.subplot(2, 3, 5)
    subplt5.set_title('new_hist_norm')
    plt.plot(sm_hist_norm)
    # norm_sum1 = sum1 - sum1.mean() / sum1.std()
    # norm_sum1 = sum1 / np.linalg.norm(sum1)
    t4 = time.time()
    print('T={:.4f}'.format(t4 - t3))
    # 计算峰
    subplot4 = plt.subplot(2, 3, 4)
    subplot4.set_title('peaks')
    real_peaks = find_real_peaks(hist_norm)
    # plt.plot(min_ext_X, min_ext_Y, 'ro')
    # plt.plot(max_exts, max_ext_Y, 'o')
    # plt.plot(peaks, hist_norm[peaks], "x")
    plt.plot(real_peaks, hist_norm[real_peaks], 'ro')
    plt.plot(hist_norm)
    # plt.vlines(x=max_exts, ymin=contour_width_heights, ymax=hist_norm[max_exts])
    # plt.hlines(*results_half[1:], color='C3')

    # plt.savefig(file_path.replace('.jpg', '.png'), dpi=600)
    # plt.close('all')  # 关闭图 0

    plt.show()


if __name__ == '__main__':
    # cumsum
    # cdf = hist_norm.cumsum()    # 定积分
    # max_exts=[3, 15, 24, 48, 66, 70, 81, 86, 99]
    # contour_width_heights=[0.34999612, 0.00083516, 0.00237699, 0.05968179, 0.00199153, 0.00391883, 0.0004497, 0.0008994, 0.00051394]
    # cumsumwlen

    # rootDir = '/disk_workspace/test_images/支柱角钢1021/角钢构件背面正面混合'
    rootDir = '/disk_workspace/test_images/支柱角钢1021/左正面'
    dd_list = ['C_ZZ_JG_1144_1833_1349_2070.jpg', 'C_ZZ_JG_4762_3885_4989_4108.jpg', 'C_ZZ_JG_4592_250_4917_603.jpg',
               'C_ZZ_JG_6037_2721_6329_3008.jpg', 'C_ZZ_JG_5669_801_6076_1220.jpg', 'C_ZZ_JG_1162_2052_1385_2287.jpg',
               'C_ZZ_JG_1192_1370_1447_1629.jpg', 'C_ZZ_JG_1202_1802_1432_2039.jpg', 'C_ZZ_JG_1218_550_1473_767',
               'C_ZZ_JG_4976_3625_5275_3887.jpg', 'C_ZZ_JG_5124_3651_5344_3876.jpg', 'C_ZZ_JG_6037_2721_6329_3008.jpg',
               ]

    # C_ZZ_JG_2312_2778_2547_2994
    small_images = ['C_ZZ_JG_6037_2721_6329_3008.jpg', ]
    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:

            if '.jpg' not in f or 'C_ZZ_JG' not in f:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f))
            im_path = os.path.join(rootDir, f)
            hist_argrelextrema(im_path)
