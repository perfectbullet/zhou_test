#!/usr/bin/python
# coding=utf-8
"""
物体与背景分割
"""

import numpy as np

import cv2
from scipy import ndimage as ndi

from skimage import morphology
from skimage.color import label2rgb

_DEBUG = True

# 测试代码
wind_source = 'source-img'
cv2.namedWindow(wind_source, cv2.WINDOW_NORMAL)
cv2.moveWindow(wind_source, 0, 0)

wind_elevation = 'elevation-img'
cv2.namedWindow(wind_elevation, cv2.WINDOW_NORMAL)
cv2.moveWindow(wind_elevation, 500, 0)

wind_markers = 'markers-img'
cv2.namedWindow(wind_markers, cv2.WINDOW_NORMAL)
cv2.moveWindow(wind_markers, 1000, 0)

wind_segment = 'segment-img'
cv2.namedWindow(wind_segment, cv2.WINDOW_NORMAL)
cv2.moveWindow(wind_segment, 500, 500)


# wind_cany = 'cany-img'
# cv2.namedWindow(wind_cany, cv2.WINDOW_NORMAL)
# cv2.moveWindow(wind_cany, 500, 1000)


def segment(gray_img):
    if _DEBUG:
        cv2.imshow(wind_source, gray_img)

    ######################################################################
    '''
    We therefore try a region-based method using the watershed transform. 
    First, we find an elevation map using the Sobel gradient of the image.
    '''
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # Gradient-X
    # grad_x = cv.Scharr(gray,ddepth,1,0)
    grad_x = cv2.Sobel(gray_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    ## [convert]
    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    ## [convert]
    ## [blend]
    ## Total Gradient (approximate)
    elevation_map = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [blend]

    # elevation_map = sobel(gray_img)
    if _DEBUG:
        cv2.imshow(wind_elevation, elevation_map)

    ######################################################################
    # Next we find markers of the background and the coins based on the extreme parts of the histogram of gray values.
    markers = np.zeros_like(gray_img)
    markers[gray_img < 15] = 1
    markers[gray_img > 20] = 2
    if _DEBUG:
        cv2.imshow(wind_markers, markers)

    ######################################################################
    # Finally, we use the watershed transform to fill regions of the elevation
    # map starting from the markers determined above:

    segmentation = morphology.watershed(elevation_map, markers)
    if _DEBUG:
        cv2.imshow(wind_markers, segmentation)

    ######################################################################
    # This last method works even better, and the coins can be segmented and labeled individually.

    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_coins, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_coins, image=gray_img)

    if _DEBUG:
        cv2.imshow(wind_segment, labeled_coins)
        cv2.waitKey(0)


if __name__ == '__main__':
    from os.path import join

    # 测试接口#############################################
    rootDir = '/disk_workspace/test_images/C_ZZJG0929/pickledata'
    import os
    from os import walk
    from data.utils import loadPicklePath
    from data.bugtype import TBugType
    from data.datadb import gDataDB

    # print(os.path.abspath(os.curdir))
    gDataDB.reset()
    items = gDataDB.getBugType()
    tbugtype = TBugType(items)

    for root, dirs, files in walk(rootDir):
        files = (join(root, name) for name in files)
        for f_name in files:
            # for tf in test_file_list:
            #     if tf in f:
            if '.txt' not in f_name:
                continue
            print('\n\n{}- filename: {}'.format('*' * 60, f_name))
            task = loadPicklePath(f_name)
            task.load()
            task.xml.removeDupAreas(0.2)
            compAreas = task.xml.getAreasByFlag('C_')  # 获取名称以C_开头的，cArea区域
            badareas = []
            for comp in compAreas:
                if 'C_ZZ_JG' in comp.name:
                    jg_img = cv2.cvtColor(task.img.getImgROI(comp), cv2.COLOR_BGR2GRAY)
                    segment(jg_img)
