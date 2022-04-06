import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show

from utils.consts import HORIZONTAL_WINDOW, INPUT_IMAGE_PATH, VERTICAL_WINDOW, PREWITT_WINDOW_MASK1, \
    PREWITT_WINDOW_MASK2, \
    WINDOW_LAPLACIAN_AGREEMENT_METHOD, LAPLACIAN_WINDOW, LAPLACIAN_WINDOW_PERPENDICULAR, \
    LAPLACIAN_WINDOW_APPROXIMATIONS
from utils.histogram import create_figure_of_gradient_method
from utils.images import open_image_as_arrays
from utils.window import make_convolve, gradient_module, border_processing, make_convolve_and_show


def make_convolve_and_show_gradient(img_array, window1, window2):
    s1 = make_convolve(img_array, window1)
    s2 = make_convolve(img_array, window2)
    img_gradient = gradient_module(s1, s2)

    create_figure_of_gradient_method(img_array,
                                     np.abs(s1),
                                     img_gradient,
                                     np.abs(s2),
                                     border_processing(img_gradient))
    plt.tight_layout()
    show()


image_filepath = INPUT_IMAGE_PATH
img_as_array = open_image_as_arrays(image_filepath)


# a = np.array([[0, 1, 2, 4, 7, 10, 15],
#               [0, 1, 2, 4, 7, 10, 15],
#               [0, 1, 2, 4, 7, 10, 15],
#               [80, 70, 60, 50, 40, 30, 20],
#               [80, 70, 60, 50, 40, 30, 20]])
# print(a)
# grad = signal.convolve2d(a, HORIZONTAL_WINDOW, boundary='symm', mode='same')
# print("Convolve result")
# print(grad)
# print("Another")
# print(a)
# print("Convolve result")
# grad = signal.convolve2d(a, VERTICAL_WINDOW, boundary='symm', mode='same')
# print(grad)
# print(np.abs(grad))


make_convolve_and_show_gradient(img_as_array, HORIZONTAL_WINDOW, VERTICAL_WINDOW)
make_convolve_and_show_gradient(img_as_array, PREWITT_WINDOW_MASK1, PREWITT_WINDOW_MASK2)

make_convolve_and_show(img_as_array, WINDOW_LAPLACIAN_AGREEMENT_METHOD)
make_convolve_and_show(img_as_array, LAPLACIAN_WINDOW)
make_convolve_and_show(img_as_array, LAPLACIAN_WINDOW_PERPENDICULAR)
make_convolve_and_show(img_as_array, LAPLACIAN_WINDOW_APPROXIMATIONS)