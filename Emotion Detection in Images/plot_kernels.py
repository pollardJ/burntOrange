# Authors: David Wen, Alex Romriell, Jacob Pollard

import time
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.io import imshow, show
from scipy.signal import convolve2d
import numpy as np
from pretrained_model import get_our_images


IMDIR = "~/our_images/"
# KERNELS = "~/large_3072_rescale_disgust_sad.npz"
KERNELS = "four_cnn_fc_4096_rescale_happy_disgust.npz"
# KERNELS = "large_4096_rescale_happy_sad.npz"
# SAVEPATH = "kernelgrid.png"
# SAVEPATH = "kernelgrid_present.png"
# SAVEPATH = "kernelgrid_color.png"

C1 = {'cmap': 'Greys'}
C2 = {'cmap': 'Greys_r'}
C3 = {'cmap': 'rainbow'}

K_NUMS = [0, 9, 11, 15, 16, 18, 21, 23, 26, 30]

def filtered_image(img, ker):

    return convolve2d(img, ker, mode="valid", boundary="fill")


if __name__ == "__main__":

    start = time.time()

    originals, our_images, _ = get_our_images(IMDIR)

    print np.shape(our_images)

    # get our pretrained parameters
    with np.load(KERNELS) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    for i in range(len(param_values)):
        print np.shape(param_values[i])

    rows = np.shape(our_images)[0]
    cols = len(K_NUMS)

    fig, axarr = plt.subplots(rows, cols + 1)

    for r in range(rows):

        axarr[r, 0].imshow(originals[r, 0], vmin=0, vmax=1, **C2)
        axarr[r, 0].grid(False)

    for c in range(1, cols + 1):

        for r in range(rows):

            conv_img = filtered_image(our_images[r, 0], param_values[6][K_NUMS[c - 1], 0])
            axarr[r, c].imshow(conv_img, **C3)
            axarr[r, c].grid(False)

    # remove the axis tick marks
    for r in range(rows):

        plt.setp([a.get_xticklabels() for a in axarr[r, :]], visible=False)

    for c in range(cols + 1):

        plt.setp([a.get_yticklabels() for a in axarr[:, c]], visible=False)


    # save the figure
    fig.savefig(SAVEPATH, dpi=500, bbox_inches="tight")


    print "This took {} seconds to run...".format(time.time() - start)


