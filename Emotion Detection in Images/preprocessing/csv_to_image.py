# Authors: D. Wen, A. Romriell, J. Pollard

"""
This file reads in the image data for the Facial Expression Recognition data as a .csv file
and converts the pixel data into picture files in .png format sized 48 x 48

You only need to specify your FILEPATH to the fer2013.csv location and the OUTPATH location where
you want the files to be saved. Note that OUTPATH does not need a trailing backslash
"""

import time
import numpy as np
from scipy.misc import imsave



# where the .csv file is at and where you want the images to go
FILEPATH = "fer2013.csv"
OUTPATH = "training_images"


OUTTEST = "fer_python_test"

# contains the emotion codes for the images
# A=Angry, D=Disgust, F=Fear, H=Happy, U=Unhappy, S=Surprise, N=Neutral
EMOTIONS = {'0': 'A', '1': 'D', '2': 'F', '3': 'H', '4': 'U', '5': 'S', '6': 'N'}



def get_data(filepath):
    """
    -opens the file in filepath and returns a list of strings that are each line in the file
    :param filepath:
    :return:
    """

    with open(filepath) as fl:

        data = [line for line in fl]

    return data


def split_convert_pixels(pix_string):
    """
    -parses a long string of pixel values into a list of integers
    :param pix_string:
    :return:
    """
    pixel_values = list()
    pixels = pix_string.split(" ")

    for pix in pixels:

        pixel_values.append(int(pix))

    return pixel_values


def pixelList_to_pixelMatrix(pixel_list):
    """
    -accepts a list of pixels and parses it to return a pixel matrix
    -note that the images are 48 X 48 according to the readme file
    :param pixel_list:
    :return:
    """
    factors = range(1, 49)
    pixel_matrix = list()

    for f in factors:

        pixel_matrix.append(pixel_list[(f-1)*48:f*48])

    return np.array(pixel_matrix, np.uint8)



def parse_data(data_list):
    """
    -accepts a list of data of the form "emotion,num num ...,usage" and returns
    a list of dictionaries of the form {"emotion": number btw 0 and 6, "pixels": list of pixels, "usage": training/test}
    :param data_list:
    :return:
    """
    parsed_data = list()

    print "Skipping the first element because it just contains the column headers..."

    for i in range(1, len(data_list)):

        if i == len(data_list)/2:
            print "\n'Suh dude. Halfway done...\n"

        img_dict = dict()
        pieces = data_list[i].split(',')
        img_dict['emotion'] = pieces[0]
        img_dict['pixels'] = pixelList_to_pixelMatrix(split_convert_pixels(pieces[1]))
        img_dict['usage'] = pieces[2].strip('\n')
        parsed_data.append(img_dict)

    return parsed_data


def pixels_to_images(data_dicts, emo_dict, dir):
    """
    -takes in the dictionary with pixel and emotion classes, the dictionary with emotion codes,
    an an outpath directory
    -saves each image as a .png file with the naming convention "EmotionCode000000.png"
    :param data_dict:
    :param dir:
    :return:
    """
    pic_num = 1

    for image in data_dicts:

        filename = dir + '/' + emo_dict[image['emotion']] + '{:06d}'.format(pic_num) + '.png'
        imsave(filename, image['pixels'], format='png')
        pic_num += 1


if __name__ == "__main__":

    start = time.time()

    print "\nGetting the lines of the file into a list to be parsed...\n"
    csvs = get_data(FILEPATH)

    print "\nThere are {0} images in the file...\n".format(len(csvs)-1)

    print "\nParsing the data...\n"
    # returns a list of dictionaries of the form
    # {'emotion': emotion code, 'usage': usage type, 'pixels': list of pixels}
    parsed_data = parse_data(csvs)

    print "\nThe data for {0} images has been parsed...\n".format(len(parsed_data))
    # print "The shape of the pixel matrix is {0}".format(np.shape(parsed_data[0]['pixels']))
    # print parsed_data[0]['emotion']


    # For testing the file conversion/saving on a small set of image data
    # subset_parsed = parsed_data[0:9]

    print "\nSaving the pixel matrices as .png files to {0} ...".format(OUTPATH)
    pixels_to_images(parsed_data, EMOTIONS, OUTPATH)

    print "\nFinished saving the {0} images to {1} ...".format(len(parsed_data), OUTPATH)
    print "\nThanks for stopping by. Come back and see us soon, okay."

    print "\nThis took {0:.4f} seconds to process...\n".format(time.time()-start)





