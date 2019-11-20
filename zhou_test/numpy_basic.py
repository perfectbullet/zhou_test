import numpy as np

if __name__ == '__main__':
    b = (255 - 0) * np.random.random((4, 5, 3)) + 0
    print(b)
    print('')
    print(b[:, :, 0:1])
