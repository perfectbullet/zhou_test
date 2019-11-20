#!/usr/bin/python
# coding=utf-8

import cv2 as cv
import numpy as np

SZ = 20
bin_n = 16  # Number of bins


affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

## [deskew] 偏斜校正
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    # print('m[mu02]={}, m[mu11]={}, skew={}'.format(m['mu02'], m['mu11'], skew))
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img
# # [deskew]


# # [hog]
def hog(img):
    # img.shape is (20, 20)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)   # 计算2D向量的大小和角度。   mag.shape=(20, 20), ang.shape=(20, 20)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)  量化二元值   bins.shape=(20, 20)
    # 分为 4 个区域，
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    # bincount(x, weights=None, minlength=0)
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector, 梯度直方图
    # print('mag.shape={}, ang.shape={}, bins.shape={}, bin_cells[0].shape={}, hists[0].shape={}, hist.shape={}'
    #       .format(mag.shape, ang.shape, bins.shape, bin_cells[0].shape, hists[0].shape, hist.shape))

    # mag.shape=(20, 20), ang.shape=(20, 20), bins.shape=(20, 20), bin_cells[0].shape=(10, 10),
    # hists[0].shape=(16,), hist.shape=(64,)
    return hist
## [hog]


img = cv.imread('digits.png', 0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")


cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# First half is trainData, remaining is testData
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

# #####     Now training      ########################
deskewed = [list(map(deskew, row)) for row in train_cells]
hogdata = [list(map(hog, row)) for row in deskewed]     # len(hogdata)=50, len(hogdata[0]=50, hogdata[0][0].shape=(64,)
trainData = np.float32(hogdata).reshape(-1, 64)     # trainData.shape=(2500, 64)
responses = np.repeat(np.arange(10), 250)[:, np.newaxis]    # responses.shape=(2500, 1), 相当于标签
print('len(hogdata)={}, len(hogdata[0]={}, hogdata[0][0].shape={}, trainData.shape={}, responses.shape={}'
      .format(len(hogdata), len(hogdata[0]), hogdata[0][0].shape, trainData.shape, responses.shape))

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

######     Now testing      ########################

deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
