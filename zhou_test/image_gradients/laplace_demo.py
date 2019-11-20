"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import cv2 as cv
import numpy as np

dst_std = []


def main(img_path):
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_64F
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    # imageName = argv[0] if len(argv) > 0 else 'lena.jpg'

    src_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)     # Load an image
    # src_gray = cv.resize(src_gray, (128, 128), interpolation=cv.INTER_AREA)
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src_gray = cv.GaussianBlur(src_gray, (3, 3), 0)
    # [reduce_noise]

    # Create Window
    # cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, kernel_size)
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    # [convert]

    # [display]
    # cv.imshow(window_name, abs_dst)
    # cv.waitKey(0)
    # [display]

    # return 0

    # print('dst.shape={}, dst.mean={}, dst.var={}, dst.std={}.'.format(dst.shape, dst.mean(), dst.var(), dst.std()))
    # print('abs_dst.shape={}, abs_dst.mean={}, abs_dst.var={}, abs_dst.std={} \n\n'.format(abs_dst.shape, abs_dst.mean(),
    #                                                                                       abs_dst.var(), abs_dst.std()))
    if dst.std() > 0:
        print(img_path)
        print('dst.shape={}, dst.mean={:.5f}, dst.var={:.3f}, dst.std={:.3f}.'.format(dst.shape, dst.mean(), dst.var(), dst.std()))
        print('abs_dst.shape={}, abs_dst.mean={:.5f}, abs_dst.var={:.3f}, abs_dst.std={:.3f} \n\n'.format(abs_dst.shape, abs_dst.mean(),
                                                                                              abs_dst.var(), abs_dst.std()))
        dst_std.append(dst.std())


if __name__ == "__main__":
    # img_path = '/disk_workspace/train_data_for_svm/dwhxj_train_four_class/C_DWHXJ_FM_MH/2-708_1024_753.jpg'
    # clear_path = '/disk_workspace/train_data_for_svm/dwhxj_train_four_class/C_DWHXJ_FM/2-1394_5479_994.jpg'
    # main(img_path)
    # main(clear_path)

    import os
    mh_img_dir = '/disk_workspace/train_data_for_svm/dwhxj_train_four_class/C_DWHXJ_ZM_MH'
    # dst_std: len=124, mean=1.28457109415, max=4.03412184699, min=0.861563168357, median=1.14827293991

    qx_img_dir = '/disk_workspace/train_data_for_svm/dwhxj_train_four_class/C_DWHXJ_ZM'
    # dst_std: len=1012, mean=2.38522690802, max=8.23299568756, min=0.852506280799, median=2.00159004353
    for im_name in os.listdir(qx_img_dir):
        if im_name.endswith('.jpg'):
            main(os.path.join(qx_img_dir, im_name))
    print('dst_std: len={}, mean={}, max={}, min={}, median={}'
          .format(len(dst_std), np.mean(dst_std), np.max(dst_std), np.min(dst_std), np.median(dst_std)))

