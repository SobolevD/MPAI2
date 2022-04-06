from skimage.io import imsave, imread


def open_image_as_arrays(filepath):
    return imread(filepath)


def save_image(img, directory):
    imsave(directory, img)