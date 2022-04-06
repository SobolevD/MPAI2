import json

import numpy as np
from scipy import signal
from skimage.io import show
from matplotlib import pyplot as plt

from utils.consts import WINDOW_HORIZONTAL, INPUT_IMAGE_PATH, WINDOW_VERTICAL, WINDOW_PREWITT_S1, WINDOW_PREWITT_S2, \
    WINDOW_LAPLACIAN_AGREEMENT_METHOD, WINDOW_LAPLACIAN, WINDOW_LAPLACIAN_MUTUALLY_PERPENDICULAR, \
    WINDOW_LAPLACIAN_OF_SUM_APPROXIMATIONS
from utils.histogram import create_figure_of_gradient_method, create_figure_of_laplacian_method
from utils.images import open_image_as_arrays
from utils.window import window_processing, gradient_module, border_processing

image_filepath = INPUT_IMAGE_PATH
img_as_array = open_image_as_arrays(image_filepath)


a = np.array([[0, 1, 2, 4, 7, 10, 15], [0, 1, 2, 4, 7, 10, 15], [0, 1, 2, 4, 7, 10, 15], [80, 70, 60, 50, 40, 30, 20], [80, 70, 60, 50, 40, 30, 20]])
print(a)
grad = signal.convolve2d(a, WINDOW_HORIZONTAL, boundary='symm', mode='same')
print("Convolve result")
print(grad)
print("Another")
print(a)
print("Convolve result")
grad = signal.convolve2d(a, WINDOW_VERTICAL, boundary='symm', mode='same')
print(grad)
print(np.abs(grad))
print("New age begins")


img_derivative_horizontal = window_processing(img_as_array, WINDOW_HORIZONTAL)
img_derivative_vertical = window_processing(img_as_array, WINDOW_VERTICAL)
img_gradient = gradient_module(img_derivative_horizontal, img_derivative_vertical)

create_figure_of_gradient_method(img_as_array, np.abs(img_derivative_horizontal), img_gradient, np.abs(img_derivative_vertical), border_processing(img_gradient))
plt.tight_layout()
show()

img_prewitt_s1 = window_processing(img_as_array, WINDOW_PREWITT_S1)
img_prewitt_s2 = window_processing(img_as_array, WINDOW_PREWITT_S2)
img_gradient_prewitt = gradient_module(img_prewitt_s1, img_prewitt_s2)

create_figure_of_gradient_method(img_as_array, np.abs(img_prewitt_s1), img_gradient_prewitt, np.abs(img_prewitt_s2), border_processing(img_gradient_prewitt))
plt.tight_layout()
show()


img_laplacian_agreement = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_AGREEMENT_METHOD))
create_figure_of_laplacian_method(img_as_array, img_laplacian_agreement, border_processing(img_laplacian_agreement))
plt.tight_layout()
show()


img_laplacian = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN))
create_figure_of_laplacian_method(img_as_array, img_laplacian, border_processing(img_laplacian))
plt.tight_layout()
show()


img_laplacian_mutually_perpendicular = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_MUTUALLY_PERPENDICULAR))
create_figure_of_laplacian_method(img_as_array, img_laplacian_mutually_perpendicular, border_processing(img_laplacian_mutually_perpendicular))
plt.tight_layout()
show()


img_laplacian_of_sum_approximations = np.abs(window_processing(img_as_array, WINDOW_LAPLACIAN_OF_SUM_APPROXIMATIONS))
create_figure_of_laplacian_method(img_as_array, img_laplacian_of_sum_approximations, border_processing(img_laplacian_of_sum_approximations))
plt.tight_layout()
show()