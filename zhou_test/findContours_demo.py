from __future__ import print_function

import argparse
import numpy as np
import random as rng

import cv2 as cv

rng.seed(12345)


def thresh_callback(val):
    threshold = val

    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
    _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # contours mask
    mask = np.zeros(src_gray.shape, np.uint8)
    cv.drawContours(mask, contours, -1, 255, -1)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 1, cv.LINE_8, hierarchy, 0)

    pixelpoints = np.transpose(np.nonzero(mask))
    # pixelpoints = cv2.findNonZero(mask)

    # Show in a window
    cv.namedWindow('Contours', cv.WINDOW_NORMAL)
    cv.imshow('Contours', drawing)
    cv.namedWindow('canny_output', cv.WINDOW_NORMAL)
    cv.imshow('canny_output', mask)


# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
args = parser.parse_args()

img_file = '/disk_workspace/test_images/test_kongdong/gray_images/12.jpg'
# src = cv.imread(cv.samples.findFile(args.input))
src = cv.imread(img_file)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 50  # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
