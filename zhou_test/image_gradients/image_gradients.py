import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def show(img_path, src_img, laplacian, abs_lap64f):
    lap_txt = 'laplacian shape={} type={} std={:.2f}\n mean={:.2f} max={} min={}'\
            .format(laplacian.shape,  laplacian.dtype,  laplacian.std(), laplacian.mean(), laplacian.max(), laplacian.min())
    lap64_txt = 'abs_lap64f shape={} type={} std={:.2f}\n mean={:.2f} max={} min={}'\
        .format(abs_lap64f.shape,  abs_lap64f.dtype,  abs_lap64f.std(), abs_lap64f.mean(), abs_lap64f.max(), abs_lap64f.min())
    
    plt.subplot(2, 2, 1), plt.imshow(src_img, cmap='gray')
    plt.title('Original', fontsize=6),  plt.xticks([]),  plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title(lap_txt, fontsize=6),  plt.xticks([]),  plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(abs_lap64f, cmap='gray')
    plt.title(lap64_txt, fontsize=6),  plt.xticks([]),  plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(abs_lap64f, cmap='gray')
    # plt.title('abs_lap64f'), plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(img_path.replace('.jpg', '.png'), dpi=500)
    plt.close()


def main(img_path):
    gray_img = cv.imread(img_path, 0)
    laplacian = cv.Laplacian(gray_img, cv.CV_64F)
    abs_lap64f = np.absolute(laplacian)
    show(img_path, gray_img, laplacian, abs_lap64f)


def list_images(dir_path):
    pass


if __name__ == '__main__':
    """
    laplacian shape=(357, 313),  type=float64,  std=10.7155648791, mean=0.00136028852436, max=182.0, min=-159.0
    abs_lap64f shape=(357, 313),  type=float64,  std=9.46642548953, mean=5.02096813166, max=182.0, min=0.0
    """
    clear_img_path = 'clear.jpg'
    '''
    laplacian shape=(238, 212),  type=float64,  std=2.0830761685, mean=0.00033692722372, max=36.0, min=-99.0
    abs_lap64f shape=(238, 212),  type=float64,  std=1.93851787909, mean=0.762466307278, max=99.0, min=0.0
    '''
    blurry_img_path = 'blurry.jpg'
    
    main(clear_img_path)
    main(blurry_img_path)
