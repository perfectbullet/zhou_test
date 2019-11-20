#!/usr/bin/python
# coding=utf-8
"""
==================================================
Comparing edge-based and region-based segmentation
==================================================

In this example, we will see how to segment objects from a background. We use
the ``coins`` image from ``skimage.data``, which shows several coins outlined
against a darker background.
物体与背景分割
"""

import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.feature import canny

_DEBUG = True


# 测试代码
# wind_source = 'source-img'
# cv2.namedWindow(wind_source, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_source, 0, 0)
#
# wind_elevation = 'elevation-img'
# cv2.namedWindow(wind_elevation, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_elevation, 500, 0)
#
#
# wind_markers = 'markers-img'
# cv2.namedWindow(wind_markers, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_markers, 1000, 0)
#
# wind_segment = 'segment-img'
# cv2.namedWindow(wind_segment, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_segment, 500, 500)
#
# wind_cany = 'cany-img'
# cv2.namedWindow(wind_cany, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_cany, 500, 1000)


def segment(file_path):
    ######################################################################
    # Edge-based segmentation
    # =======================
    #
    # Next, we try to delineate the contours of the coins using edge-based
    # segmentation. To do this, we first get the edges of features using the
    # Canny edge-detector.

    img = cv2.imread(file_path, flags=0)
    com_name, part_name = os.path.basename(file_path)[:-4].split('-')
    com_idx = com_name.split('_')[-4:]
    com_idx[2:] = com_idx[:2]
    part_idx = part_name.split('_')[-4:]
    # part_idx = map(lambda x, y: int(x) - int(y), part_idx, com_idx)
    # part_idx = tuple(part_idx)      # 图片下标

    edges = canny(img)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('Canny detector')
    ax.axis('off')

    ######################################################################
    # These contours are then filled using mathematical morphology.

    fill_coins = ndi.binary_fill_holes(edges)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('filling the holes')
    ax.axis('off')

    ######################################################################
    # Small spurious objects are easily removed by setting a minimum size for valid objects.

    from skimage import morphology

    coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('removing small objects')
    ax.axis('off')

    plt.tight_layout()
    # plt.show()

    plt.savefig(file_path.replace('.jpg', '.png'), dpi=300)
    plt.close('all')  # 关闭图 0


if __name__ == '__main__':
    from os.path import join

    # 测试接口#############################################
    rootDir = '/disk_workspace/xiaozhang-out/C_TGSE_A'
    import os
    from os import walk
    from data.utils import loadPicklePath
    from data.bugtype import TBugType
    from data.datadb import gDataDB

    # gDataDB.reset()
    # items = gDataDB.getBugType()
    # tbugtype = TBugType(items)

    # for root, dirs, files in walk(rootDir):
    #     files = [join(root, name) for name in files]
    #     for f in files:
    #         # for tf in test_file_list:
    #         #     if tf in f:
    #         if '吐库二线_下行_珍珠泉至托克逊__1095_62.291.txt' not in f:
    #             continue
    #         print('\n\n{}- filename: {}'.format('*' * 60, f))
    #         task = loadPicklePath(os.path.join(rootDir, f))
    #         task.load()
    #         task.xml.removeDupAreas(0.2)
    #         compAreas = task.xml.getAreasByFlag('C_')  # 获取名称以C_开头的，cArea区域
    #         badareas = []
    #         for comp in compAreas:
    #             if 'C_ZZ_JG' in comp.name:
    #                 jg_img = cv2.cvtColor(task.img.getImgROI(comp), cv2.COLOR_BGR2GRAY)
    #                 segment(jg_img)

    for root, dirs, files in walk(rootDir):
        files = [join(root, name) for name in files]
        for f in files:
            # for tf in test_file_list:
            #     if tf in f:
            if '.jpg' not in f:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f))
            img = cv2.imread(f)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            segment(f)