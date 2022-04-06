import numpy as np
from skimage.io import imshow
from matplotlib import pyplot as plt


def create_figure_of_gradient_method(src_img, deriv_horiz, grad_matrix, deriv_vert, img_border):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Source image")
    imshow(src_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title("Derivative horizontal")
    imshow(deriv_horiz, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 3)
    plt.title("Gradient evaluation")
    imshow(grad_matrix, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("Border processing")
    imshow(img_border, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title("Derivative vertical")
    imshow(deriv_vert, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 6)
    plt.title("Histogram of gradient evaluation")
    plt.xlabel('Brightness values')
    plt.ylabel('Pixels quantity')
    create_wb_histogram_plot(grad_matrix)
    return fig


def create_figure_of_laplacian_method(src_img, laplacian, img_border):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    plt.title("Source image")
    imshow(src_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 2)
    plt.title("Laplacian evaluation")
    imshow(laplacian, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 3)
    plt.title("Border processing")
    imshow(img_border, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 4)
    plt.title("Histogram of laplacian evaluation")
    plt.xlabel('Brightness values')
    plt.ylabel('Pixels quantity')
    create_wb_histogram_plot(laplacian)
    return fig

def create_wb_histogram_plot(img_as_arrays):
    hist, bins = np.histogram(img_as_arrays.flatten(), 256, [0, 256])
    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)
