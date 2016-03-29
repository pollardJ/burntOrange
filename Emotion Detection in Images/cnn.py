# Authors: D. Wen, A. Romriell, J. Pollard

import matplotlib 
matplotlib.use('Agg') 

import logging
import time
import random
from glob import glob
from skimage.io import imread
from skimage.util import crop
import numpy as np
import theano
import theano.tensor as T
import lasagne
import ConfigParser


def printAndLog(msg):
    print msg
    logging.info(msg)


def get_emo_int(filename):
    """
    -returns an integer between 0 and 6, inclusive, to identify one of the six emotions
    :param filename:
    :return:
    """
    pieces = filename.split('/')
    emo_key = pieces[-1][0]

    return emotions[emo_key]


def get_image_files(im_dir):
    """
    -load in the images from the user specified directory
    :param tr_dir:
    :param te_dir:
    :return:
    """
    random.seed(time.time())
    image_filenames = glob('{}/*.png'.format(im_dir))
    image_filenames.sort()
    random.shuffle(image_filenames)

    return image_filenames


def rescale_image(image):

    rows, columns = np.shape(image)
    rescale = [[float(image[r, c]) / 255 for c in range(columns)] for r in range(rows)]

    return np.array(rescale)


def normalize_image(image):

    rows, columns = np.shape(image)
    m = np.mean(image)
    s = np.std(image)
    normalized = [[float(image[r, c] - m) / s for c in range(columns)] for r in range(rows)]

    return np.array(normalized)


def load_image_data(filelist, perc_tr=0.7, im_func=None):
    """
    -load in the images as numpy arrays and return the following:
    1. a numpy array of shape (perc_tr*len(filelist), 1, 48, 48) that contains the training image data
    2. a numpy array of shape (perc_tr*len(filelist),) that contains the training emotion code for each training image
    3. a numpy array of shape ((1-perc_tr)*len(filelist), 1, 48, 48) that contains the test image data
    4. a numpy array of shape ((1-perc_tr)*len(filelist), ) that contains the test emotion codes for each test image
    :param filelist:
    :return:
    """
    if perc_tr < 0.5:
        print "Training subset percent too low, defaulting to 50%..."
        perc_tr = 0.5

    # subset the list of file names into training and testing
    # note that the file names have been shuffled randomly before coming
    # in to the function
    tr_len = int(round(perc_tr * len(filelist)))
    te_len = len(filelist) - tr_len
    training = filelist[:tr_len]
    test = filelist[tr_len:]

    # initialize empty arrays to hold the training and testing image data
    # we are cropping the images to 42 X 42
    imgs_train = np.empty(shape=(tr_len, 1, 42, 42), dtype=np.float32)
    emos_train = np.empty(shape=tr_len, dtype=np.int8)
    imgs_test = np.empty(shape=(te_len, 1, 42, 42), dtype=np.float32)
    emos_test = np.empty(shape=te_len, dtype=np.int8)

    # populate the training arrays
    for i in range(len(training)):
        img = imread(training[i], as_grey=True)
        img_crop = crop(img, crop_width=3)
        img_crop = im_func(img_crop)
        imgs_train[i, 0] = img_crop
        emos_train[i] = get_emo_int(training[i])

    # populate the test arrays
    for i in range(len(test)):
        img = imread(test[i], as_grey=True)
        img_crop = crop(img, crop_width=3)
        img_crop = im_func(img_crop)
        imgs_test[i, 0] = img_crop
        emos_test[i] = get_emo_int(test[i])

    return imgs_train, emos_train, imgs_test, emos_test


# beginning the convolutional neural network
def build_cnn(input_var=None):

    # input layer is 42 X 42 image
    network = lasagne.layers.InputLayer(shape=(None, 1, 42, 42), input_var=input_var)

    # conv1 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(c1_filter, c1_filter),
                                         stride=c1_stride, pad=c1_pad, W=lasagne.init.GlorotUniform(),
                                         nonlinearity=lasagne.nonlinearities.rectify)

    # max pool layer 1
    if (p1_pad_type=='all'):
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p1_size, p1_size), stride=p1_stride, pad=p1_pad)
    else:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p1_size, p1_size), stride=p1_stride, pad=(p1_pad, 0))

    # conv2 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(c2_filter,c2_filter),
                                         stride=c2_stride, pad=c2_pad, W=lasagne.init.GlorotUniform(),
                                         nonlinearity=lasagne.nonlinearities.rectify)

    # max pool layer 2
    if (p2_pad_type=='all'):
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p2_size, p2_size), stride=p2_stride, pad=p2_pad)
    else:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p2_size, p2_size), stride=p2_stride, pad=(p2_pad, 0))

    # conv3 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(c3_filter, c3_filter),
                                         stride=c3_stride, pad=c3_pad, W=lasagne.init.GlorotUniform(),
                                         nonlinearity=lasagne.nonlinearities.rectify)

    # max pool layer 3
    if (p3_pad_type=='all'):
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p3_size, p3_size), stride=p3_stride, pad=p3_pad)
    else:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(p3_size, p3_size), stride=p3_stride, pad=(p3_pad, 0))


    # enter the fully connected hidden layer
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=dropout),
                                        num_units=fc_units, W=lasagne.init.GlorotUniform(),
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

    # again the num_units=7 refers to the 7 emotions we are classifying
    network = lasagne.layers.DenseLayer(network, num_units=7,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def minibatch_iterator(inputs, targets, batchsize):
    """
    -returns subsets of the input data
    -note that the function assumes the data has been randomly shuffled already
    before being passed
    :param inputs:
    :param targets:
    :param batchsize:
    :return:
    """

    num_batches = int(len(inputs)/batchsize)

    for i in range(num_batches):
        if i != num_batches - 1:
            yield inputs[i*batchsize:(i+1)*batchsize, :, :, :], targets[i*batchsize:(i+1)*batchsize, ]
        else:
            yield inputs[i*batchsize:, :, :, :], targets[i*batchsize:, ]

if __name__ == "__main__":

    start = time.time()

    config = ConfigParser.ConfigParser()
    config.read('./emotobot.cfg')

    input_path = config.get('Input', 'input_path', 0)
    csv_path = config.get('Input', 'csv_path', 0)

    emotions = {"A": 0, "D": 1, "F": 2, "H": 3, "U": 4, "S": 5, "N": 6}

    config.read('./parameters.cfg')
    num_epochs = int(config.get('Parameters', 'num_epochs'))
    parameter_set = config.get('Parameters', 'parameter_set')

    csv_path = csv_path + '/' + parameter_set + '.csv'

    c1_filter = int(config.get(parameter_set, 'c1_filter'))
    c1_stride = int(config.get(parameter_set, 'c1_stride'))
    c1_pad = int(config.get(parameter_set, 'c1_pad'))
    p1_size = int(config.get(parameter_set, 'p1_size'))
    p1_stride = int(config.get(parameter_set, 'p1_stride'))
    p1_pad = int(config.get(parameter_set, 'p1_pad'))
    p1_pad_type = config.get(parameter_set, 'p1_pad_type')
    c2_filter= int(config.get(parameter_set, 'c2_filter'))
    c2_stride= int(config.get(parameter_set, 'c2_stride'))
    c2_pad= int(config.get(parameter_set, 'c2_pad'))
    p2_size = int(config.get(parameter_set, 'p2_size'))
    p2_stride = int(config.get(parameter_set, 'p2_stride'))
    p2_pad = int(config.get(parameter_set, 'p2_pad'))
    p2_pad_type = config.get(parameter_set, 'p2_pad_type')
    c3_filter = int(config.get(parameter_set, 'c3_filter'))
    c3_stride = int(config.get(parameter_set, 'c3_stride'))
    c3_pad = int(config.get(parameter_set, 'c3_pad'))
    p3_size = int(config.get(parameter_set, 'p3_size'))
    p3_stride = int(config.get(parameter_set, 'p3_stride'))
    p3_pad = int(config.get(parameter_set, 'p3_pad'))
    p3_pad_type = config.get(parameter_set, 'p3_pad_type')
    fc_units = int(config.get(parameter_set, 'fc_units'))
    dropout = float(config.get(parameter_set, 'dropout'))

    logging.basicConfig(filename='./log/emotobot.log',level=logging.INFO,
                        format='%(levelname)s %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    logging.info("Emotobot started**************************************")

    logging.info("num_epochs = %s" % num_epochs)
    logging.info("parameter_set = %s" % parameter_set)
    logging.info("c1_filter = %s" % (c1_filter))
    logging.info("c1_stride = %s" % (c1_stride))
    logging.info("c1_pad = %s" % (c1_pad))
    logging.info("p1_size = %s" % (p1_size))
    logging.info("p1_stride = %s" % (p1_stride))
    logging.info("p1_pad = %s" % (p1_pad))
    logging.info("p1_pad_type = %s" % (p1_pad_type))
    logging.info("c2_filter = %s" % (c2_filter))
    logging.info("c2_stride = %s" % (c2_stride))
    logging.info("c2_pad = %s" % (c2_pad))
    logging.info("p2_size = %s" % (p2_size))
    logging.info("p2_stride = %s" % (p2_stride))
    logging.info("p2_pad = %s" % (p2_pad))
    logging.info("p2_pad_type = %s" % (p2_pad_type))
    logging.info("c3_filter = %s" % (c3_filter))
    logging.info("c3_stride = %s" % (c3_stride))
    logging.info("c3_pad = %s" % (c3_pad))
    logging.info("p3_size = %s" % (p3_size))
    logging.info("p3_stride = %s" % (p3_stride))
    logging.info("p3_pad = %s" % (p3_pad))
    logging.info("p3_pad_type = %s" % (p3_pad_type))
    logging.info("fc_units = %s" % fc_units)
    logging.info("dropout = %s" % dropout)

    files = get_image_files(input_path)

    X_tr, Y_tr, X_te, Y_te = load_image_data(files, perc_tr=0.8, im_func=rescale_image)

    # name the Theano input variables
    images = T.tensor4('inputs')
    emos = T.ivector('targets')

    # create the network
    emo_cnn = build_cnn(images)
    pred = lasagne.layers.get_output(emo_cnn)
    loss = lasagne.objectives.categorical_crossentropy(pred, emos)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(emo_cnn, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.004, momentum=0.9)
    test_prediction = lasagne.layers.get_output(emo_cnn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, emos)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), emos), dtype=theano.config.floatX)

    train_func = theano.function([images, emos], loss, updates=updates)
    test_func = theano.function([images, emos], [test_loss, test_acc])

    csv_file = open(csv_path, 'w')
    csv_file.write('epoch,trainloss,testloss,testacc\n')

    # begin the training epochs
    for epoch in range(num_epochs):
        
        # In each epoch, we do a full pass over the training data:
        epoch_start_time = time.time()
        train_err = 0
        train_batches = 0

        for batch in minibatch_iterator(X_tr, Y_tr, 200):
            inputs, targets = batch
            train_err += train_func(inputs, targets)
            train_batches += 1


        # And a full pass over the test data:
        test_err = 0
        test_acc = 0
        test_batches = 0

        for batch in minibatch_iterator(X_te, Y_te, 200):
            inputs, targets = batch
            err, acc = test_func(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        printAndLog("Epoch {0} of {1} took {2:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start_time))
        printAndLog("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        printAndLog("  test loss:\t\t{:.6f}".format(test_err / test_batches))
        printAndLog("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

        csv_str = '{},{:.6f},{:.6f},{:.4f}\n'.format(epoch + 1, train_err / train_batches,
                                                   test_err / test_batches, test_acc / test_batches)
        csv_file.write(csv_str)
        logging.info(csv_str)

    printAndLog("Total time: {:.4f} seconds.".format(time.time() - start))
    logging.info("Emotobot completed**************************************")

    csv_file.close()

