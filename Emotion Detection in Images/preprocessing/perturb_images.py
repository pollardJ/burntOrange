# Authors: D. Wen, A. Romriell, J. Pollard

from scipy.misc import imsave
from scipy import ndimage
import numpy as np
from skimage.exposure import adjust_sigmoid
import glob
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('../emotobot.cfg')


def get_images(filepath):
    return glob.glob(filepath + '/*.png')


def split_images(filelist):
    np.random.shuffle(filelist)
    chunks = np.array_split(filelist, 3)
    return chunks


def flip(img):
    return np.fliplr(img)


def blur(img):
    return ndimage.gaussian_filter(img, sigma=1)


def denoise(img):
    return ndimage.median_filter(img, 2.5)


def contrast_enhance(img):
    return adjust_sigmoid(img, cutoff=0.5, gain=10)


def perturb_images(chunks):
    """

    :param chunks: a list of lists.
                   Each list will be perturbed by flip, blur, and denoise functions,
                   in that order. The perturbed image is then saved in specified
                   directory.
    :return: None
    """

    # 0: flip
    # 1: blur
    # 2: denoise

    # starting file number:
    file_num = 37423

    for i in range(len(chunks)):
        for j in range(len(chunks[i])):
            img_name = chunks[i][j]
            img = ndimage.imread(img_name)
            if i == 0:  # flip
                category = img_name[-11]  # this gets the emotion category
                flipped = flip(img)
                filename = perturbed_path + "/" + category + str(file_num) + ".png"
                imsave(filename, flipped, format='png')
                # print "Flipped:", filename

            elif i == 1:  # blur
                category = img_name[-11]
                blurred = blur(img)
                filename = perturbed_path + "/" + category + str(file_num) + ".png"
                imsave(filename, blurred, format='png')
                # print "Blurred:", filename

            elif i == 2:  # denoise
                category = img_name[-11]
                denoised = denoise(img)
                filename = perturbed_path + "/" + category + str(file_num) + ".png"
                imsave(filename, denoised, format='png')
                # print "Denoised:", filename

            file_num += 1

    return

if __name__ == '__main__':
    perturbed_path = config.get('Input', 'perturbed_path', 0)

    # get a list of files
    files = get_images(perturbed_path)

    # split the list of files into 3 ~equal lists
    three_lists = split_images(files)

    # perturb each image according to which list it is in
    perturb_images(three_lists)

    # contrast enhance all images in perturbed_path (this includes new perturbed images
    files2 = get_images(perturbed_path)
    for i in range(len(files2)):
        img = ndimage.imread(files2[i])
        enhanced = contrast_enhance(img)
        imsave(files2[i], enhanced, format='png')


