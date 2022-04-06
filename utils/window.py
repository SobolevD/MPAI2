import numpy as np
from scipy import signal

from utils.consts import BORDER_PROCESSING_PARAMETER, MIN_BRIGHTNESS_VALUE, MAX_BRIGHTNESS_VALUE


def border_processing_function(element_value):
    if element_value < BORDER_PROCESSING_PARAMETER:
        return MIN_BRIGHTNESS_VALUE
    else:
        return MAX_BRIGHTNESS_VALUE


def border_processing(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(border_processing_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def gradient_module(matrix_u, matrix_v):
    sum_of_squares = np.square(matrix_u) + np.square(matrix_v)
    return np.sqrt(sum_of_squares)


def laplacian_agreement_method(alpha, beta):
    return (alpha * 2 + beta * 2).astype(int)


def window_processing(matrix, window):
    return signal.convolve2d(matrix, window, boundary='symm', mode='same').astype(int)
