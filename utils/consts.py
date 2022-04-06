import numpy as np


INPUT_IMAGE_PATH = 'images/input.tif'

MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0

BORDER_PROCESSING_PARAMETER = 20

# Windows
HORIZONTAL_WINDOW = np.array([[-1, 1]])


VERTICAL_WINDOW = np.array([[-1],
                            [1]])

# Prewitt masks
PREWITT_WINDOW_MASK1 = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]]) * (1/6.0)


PREWITT_WINDOW_MASK2 = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]]) * (1/6.0)


# Laplacian masks
WINDOW_LAPLACIAN_AGREEMENT_METHOD = np.array([[2, -1, 2],
                                              [-1, -4, -1],
                                              [2, -1, 2]]) * (1/3.0)


LAPLACIAN_WINDOW = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])


LAPLACIAN_WINDOW_PERPENDICULAR = np.array([[1, 0, 1],
                                           [0, -4, 0],
                                           [1, 0, 1]]) * (1/2.0)


LAPLACIAN_WINDOW_APPROXIMATIONS = np.array([[1, 1, 1],
                                            [1, -8, 1],
                                            [1, 1, 1]]) * (1/3.0)
