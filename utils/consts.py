import numpy as np


INPUT_IMAGE_PATH = 'images/input.tif'
MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0
BORDER_PROCESSING_PARAMETER = 20


WINDOW_HORIZONTAL = np.array([[-1, 1]])


WINDOW_VERTICAL = np.array([[-1],
                            [1]])


WINDOW_PREWITT_S1 = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]]) * (1/6)


WINDOW_PREWITT_S2 = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]]) * (1/6)


WINDOW_LAPLACIAN_AGREEMENT_METHOD = np.array([[2, -1, 2],
                                              [-1, -4, -1],
                                              [2, -1, 2]]) * (1/3)


WINDOW_LAPLACIAN = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])


WINDOW_LAPLACIAN_MUTUALLY_PERPENDICULAR = np.array([[1, 0, 1],
                                                    [0, -4, 0],
                                                    [1, 0, 1]]) * (1/2)


WINDOW_LAPLACIAN_OF_SUM_APPROXIMATIONS = np.array([[1, 1, 1],
                                                   [1, -8, 1],
                                                   [1, 1, 1]]) * (1/3)