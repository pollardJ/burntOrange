# Student: J. Pollard


import numpy as np
from skimage import io
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.color import label2rgb
from scipy import ndimage as ndi
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# used to define the margins for our image outputs
MARGINS = {"left": 0.125, "right": 0.9, "bottom": 0.1,
           "top": 0.9, "wspace": 0.2, "hspace": 0.2}


def f(x, y, data):
    """
    - function used to get the gradient of a pixel in the gradient matrix
    - the output is used to plot a 3d map of the pixel gradient since the
    3d plotting engine in matplotlib will not just plot the values of an n X n
    array as 3d heights. So, I wrote this brief workaround
    :param x:
    :param y:
    :param data:
    :return:
    """
    return data[x, y]


def segment_image(filepath):
    """
    - accepts a file path for a *.png image and computes the watershed segmentation of the image
    - saves a 2d and 3d heatmap of the gradient image
    - saves: the segmentation, the markers, the segmented image, and the labeled image
    :param filepath:
    :return:
    """
    # open the image using scikit image file reader for images and convert
    # to an numpy array
    image_array = np.asarray(io.imread(filepath, as_grey=True))

    # add plot of the original image to the plot queue
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(image_array, cmap=plt.cm.gray, interpolation='nearest')
    ax.axis('off')
    ax.set_title('Original Image')

    # compute the pixel gradient of image using the Sobel operator
    gradient_of_image = sobel(image_array)

    # add plot of the 2d heatmap of the gradient image to the plot queue
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(gradient_of_image, cmap=plt.cm.jet, interpolation='nearest')
    ax.axis('off')
    ax.set_title('2d Heatmap of Image Gradient')

    # get the dimensions of the image to use as the x and y coords of
    # the 3d heatmap of the image gradient
    width1, height1 = gradient_of_image.shape
    x = np.arange(0, width1)
    y = np.arange(0, height1)

    # convert the x's and y's into a 2d mesh of values
    X, Y = np.meshgrid(x, y)

    # apply the simple function above to get the height of the gradient at each
    # point in the X, Y mesh
    z = np.array([f(x, y, gradient_of_image) for x, y in zip(np.ravel(X), np.ravel(Y))])

    # reshape the z's so that it matches the size and shape of the X, Y arrays
    # note that we could choose either X, or Y as the reference for reshaping
    Z = z.reshape(X.shape)

    # control how fine we want the 3d map to be
    plot_options = {'rstride':2,
                    'cstride':2,
                    'linewidth':0,
                    'antialiased':False}

    # add the 3d heatmap to the plot queue
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.ocean, **plot_options)

    # create markers for where the "water" will flood in from
    markers_for_watershed = np.zeros_like(image_array)
    markers_for_watershed[image_array < 30] = 1
    markers_for_watershed[image_array > 150] = 2

    # add the plot of the markers to the plot queue
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(markers_for_watershed, cmap=plt.cm.spectral, interpolation='nearest')
    ax.axis('off')
    ax.set_title('Image Markers')

    # segment the image using the watershed function from scikit image
    segmentation = watershed(image=gradient_of_image, markers=markers_for_watershed)

    # fill in any parts that might have been missed
    segmentation = ndi.binary_fill_holes(segmentation - 1)

    # label the parts that are separated with colors and overlay that array on the
    # original image
    labeled_image, _ = ndi.label(segmentation)
    image_label_overlay = label2rgb(labeled_image, image=image_array)

    # add side-by-side plots of the segmented image and the labeled parts of the image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
    ax1.imshow(image_array, cmap=plt.cm.gray, interpolation='nearest')
    ax1.contour(segmentation, [0.5], linewidths=1.2, colors='y')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(image_label_overlay, interpolation='nearest')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')

    # set the margins using the above guidelines
    fig.subplots_adjust(**MARGINS)

    # finally show all the plots
    plt.show()


if __name__ == '__main__':

    files = ["/Users/jtpollard/MSAN/msan621/project/palmgrey.png",
             "/Users/jtpollard/MSAN/msan621/project/conangrey2.png"]

    segment_image(files[0])

    segment_image(files[1])