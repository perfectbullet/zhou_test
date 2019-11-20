import unittest
import cv2 as cv
import numpy as np
import argparse

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255
src_gray = None


def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pass

    def test_harris(self):
        filename = 'C_ZZ_JG_1079_1749_1210_1876.jpg'
        # Load source image and convert it to gray
        src = cv.imread(filename)
        global src_gray
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # Create a window and a trackbar
        cv.namedWindow(source_window)
        thresh = 200  # initial threshold
        cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
        cv.imshow(source_window, src)
        cornerHarris_demo(thresh)
        cv.waitKey()


if __name__ == '__main__':
    unittest.main()
