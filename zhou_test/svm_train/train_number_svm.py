#!/usr/bin/python
# coding=utf-8

import cv2, os
import numpy as np


# 获取数据集
# 参数: datadirs:数据目录,  labels:数据目录对应的标签, descriptor:特征描述器, size:图片归一化尺寸(通常是2的n次方, 比如(64,64)), kwargs:描述器计算特征的附加参数
# 返回值, descs:特征数据, labels:标签数据
def getDataset(datadirs, labels, descriptor, size, **kwargs):
    # 获取训练数据
    # 参数: path:图片目录,  label:图片标签,  descriptor:特征描述器, size:图片归一化尺寸(通常是2的n次方, 比如(64,64)), kwargs:描述器计算特征的附加参数
    # 返回值: 图像数据, 标签数据
    def getDatas(path, label):
        datas = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                lowname = fname.lower()
                if not lowname.endswith('.jpg') and not lowname.endswith('.png') and not lowname.endswith('.bmp'): continue
                imgpath = os.path.join(root, fname)
                gray = cv2.imread(imgpath, 0)
                if gray is None or len(gray) < 10: continue
                desc = descriptor.compute(cv2.resize(gray, size,interpolation=cv2.INTER_AREA), **kwargs).reshape((-1))
                datas.append(desc)
        return np.array(datas), np.full((len(datas)), label, dtype=np.int32)

    descs, dlabels = None, None
    for path, label in zip(datadirs,labels):
        if descs is None:
            descs, dlabels = getDatas(path, label)
        else:
            ds, ls = getDatas(path, label)
            descs, dlabels = np.vstack((descs, ds)), np.hstack((dlabels, ls))
    return descs, dlabels


if __name__ == '__main__':
    from os.path import join, basename
    from os import walk
    # 正样本的标签为1, 负样本的标签为0
    # base_train_dir = '/disk_workspace/train_data_for_svm/0-9_train/'
    base_train_dir = '/disk_workspace/train_data_for_svm/dzx_number'

    dir_ls = [dr for dr in os.listdir(base_train_dir) if not dr.endswith('.dat')]
    # train_dirs = [join(base_train_dir, d) for d in dir_ls if not d.endswith('_test')]
    # train_labels = [int(basename(d)) for d in train_dirs]
    test_dirs = [join(base_train_dir, d) for d in dir_ls if d.endswith('_test')]
    test_labels = [int(basename(d).split('_')[0]) for d in test_dirs]

    outpath = join(base_train_dir, 'digits-20191114-ten.dat')  # 模型输出目录

    # hog特征描述器
    # 参数图解: https://blog.csdn.net/qq_26898461/article/details/46786285
    # 参数说明: winSize:窗口大小, blockSize:块大小, blockStride:块滑动增量, cellSize:胞元大小, nbins:梯度方向数目
    descriptor = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    # # # 拟合
    # train_datas, train_labels = getDataset(train_dirs, train_labels, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    # print('train_datas.shape={}, train_labels.shape={}'.format(train_datas.shape, train_labels.shape))
    # svm = cv2.ml.SVM_create()
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setC(2.67)
    # svm.setGamma(5.383)
    # svm.train(train_datas, cv2.ml.ROW_SAMPLE, train_labels)
    # print('outpath={}'.format(outpath))
    # svm.save(outpath)

    # 开始测试, 测试数据和拟合的数据不能有重复, 有重复的测试结果不能说明问题
    svmer = cv2.ml.SVM_load(outpath)
    # test_dirs = [join(base_train_dir, pa) for pa in ['套管双耳上部_包含多个_test', '套管双耳下部_包含多个_test']]
    # test_lables = [1, -1]
    test_des_data, test_labels = getDataset(test_dirs, test_labels, descriptor, size=(64, 64), winStride=(8, 8), padding=(0, 0))
    test_query_data = np.array(test_des_data)  #
    ret, responses = svmer.predict(test_query_data)     # ret

    # Check Accuracy
    mask = test_labels == responses.reshape(responses.shape[0])
    correct = np.count_nonzero(mask)
    acc = correct / float(mask.size)
    print('test_labels={}, responses.shape={}, mask.shape={}, acc={}'
          .format(test_labels.shape, responses.shape, mask.shape, acc))

