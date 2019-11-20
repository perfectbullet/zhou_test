# -*- coding: utf-8 -*-
#!/usr/bin/python
#test_copyfile.py

import os, shutil
import random
from os import walk
from os.path import join


def main(img_dir, sample_rat=0.2):
    # 对根目录下的样本文件夹随机采样
    for root, dirs, files in walk(img_dir):
        if len(files) < 10:
            continue
        # random pick
        sample_files = random.sample(files, int(sample_rat * len(files)))
        print('root={}, files_len={}, sample_files_len={}'.format(root, len(files), len(sample_files)))
        test_root = root + '_test'
        # move to test dirs
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        for fname in sample_files:
            shutil.move(join(root, fname), join(test_root, fname))


if __name__ == '__main__':
    main('/disk_workspace/train_data_for_svm/dwhxj_train')
