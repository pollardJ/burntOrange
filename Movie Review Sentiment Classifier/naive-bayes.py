# Author: J. Pollard

import argparse
import re
import os
import glob
from collections import Counter
from math import log
from random import shuffle


# Stop word list
stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
             'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
             'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
             'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
             'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
             'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
             'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
             'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
             'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
             'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
             've', 'wants', 'was', 'wasn', 'we', 'were', 'what', 'when', 'where', 'which',
             'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
             'you', 'your']


def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def getFileNames(direct):
    """
    accepts a directory and returns
    a list of file names in the pos
    subdirectory and a list of file
    names in the neg subdirectory
    """
    path_neg = direct + '/neg'
    path_pos = direct + '/pos'

    neg_fls = []
    pos_fls = []

    # add pos and neg files to each list
    for infile in glob.glob(os.path.join(path_neg, '*.txt')):

        neg_fls.append(infile)

    for infile in glob.glob(os.path.join(path_pos, '*.txt')):

        pos_fls.append(infile)

    # check for empty files
    neg_fls = [f for f in neg_fls if os.stat(f).st_size != 0]
    pos_fls = [f for f in pos_fls if os.stat(f).st_size != 0]

    # return the list of neg and pos file names as a tuple
    return neg_fls, pos_fls


def scramble(filelist):
    """
    accepts s list of file names and randomly shuffles
    the indices and then returns three subsets of the
    shuffled list
    """

    # assign filelist to local variable and shuffle
    l = filelist

    shuffle(l)

    # subset the shuffled list into three lists of
    # almost equal size
    sub1 = l[0:(len(l) / 3)]
    sub2 = l[(len(l) / 3):(2 * (len(l) / 3))]
    sub3 = l[(2 * (len(l) / 3)):len(l)]

    # return the three subsets
    return sub1, sub2, sub3



def getText(filename):
    """
    accepts a file name as an argument and
    returns the content in the file as a string
    """

    infile = open(filename, 'r')
    text = infile.read()
    infile.close()

    return text


def getWords(docstr):
    """
    accepts a string of text and returns a list
    of words in the passed string
    """
    # get rid of digits and non-alphanumeric chars
    # and split on spaces
    wds = re.sub('\d', ' ', docstr)
    wds = re.sub('[\W_]', ' ', wds)
    wds = wds.split()

    # convert to lowercase and get rid of stop words
    wordlist = [w.lower() for w in wds]
    wordlist = [w for w in wordlist if w not in stopWords]
    wordlist = [w for w in wordlist if len(w) >= 3]

    return wordlist


def clsWordCounts(wd_dict, wd_list, cl):
    """
    accepts a dictionary, a list of words, and a
    class and returns an updated dictionary of word
    counts for each class
    """

    # set up counter object for words in passed word list
    word_counts = Counter(wd_list)

    # for each term in counter object
    for term in word_counts:

        # if already in dictionary add to the count
        if term in wd_dict[cl]:

            wd_dict[cl][term] += word_counts[term]

        # else add to the dictionary
        else:

            wd_dict[cl][term] = word_counts[term]

    return wd_dict


def calc_prob(wds, dic, neg, pos):
    """
    accepts a counter object of words, a dictionary of
    word counts, a number of training docs used for each class
    "pos" and "neg" and returns a tuple of log probabilities
    """
    tot = neg + pos
    p_neg = float(neg) / tot
    p_pos = float(pos) / tot
    ct_neg = sum(dic["neg"].values())
    ct_pos = sum(dic["pos"].values())
    V_neg = len(dic["neg"])
    V_pos = len(dic["pos"])
    V = V_neg + V_pos
    cstar_neg = log(p_neg)
    cstar_pos = log(p_pos)

    for term in wds:

        # if word from test doc is in training doc dictionary
        # under class "neg" compute this smoothed probability
        if term in dic["neg"]:

            p_wi_neg = float(dic["neg"][term] + 1) / (ct_neg + V + 1)

        # otherwise compute this smoothed probability
        else:

            p_wi_neg = 1.0 / (ct_neg + V + 1)

        # add to the cstar_neg variable
        cstar_neg += (wds[term] * log(p_wi_neg))

        # if word from test doc is in training doc dictionary
        # under class "pos" compute this smoothed probability
        if term in dic["pos"]:

            p_wi_pos = float(dic["pos"][term] + 1) / (ct_pos + V + 1)

        # otherwise compute this smoothed probability
        else:

            p_wi_pos = 1.0 / (ct_pos + V + 1)

        # add to the cstat_pos variable
        cstar_pos += (wds[term] * log(p_wi_pos))

    # return a tuple of the two probabilities
    return cstar_neg, cstar_pos



def classify(docs, wd_dict, cl, numneg, numpos):
    """
    accepts a list of test docs, dictionary of training
    doc word counts, a class identifier, the number of
    training docs used for each class, and returns
    the number of docs that were identified correctly
    """
    correct = 0

    for f in docs:

        wds_doc = Counter(getWords(getText(f)))

        # return tuple of log probabilities with first coordinate for the
        # prob of class neg and the second coordinate for the prob of
        # class pos
        probs = calc_prob(wds_doc, wd_dict, numneg, numpos)

        if cl == "neg":

            if probs[0] > probs[1]:

                correct += 1

        if cl == "pos":

            if probs[1] > probs[0]:

                correct += 1

    return correct


def get_results(nres, pres):
    """
    accepts two array containing the results of the
    naive Bayes algorithm and displays the results
    """
    # for the average accuracy at the end
    tot_acc = 0.0

    # for each iteration display the results
    for i in range(3):

        accuracy = float(nres[i][2]+pres[i][2]) / (nres[i][0]+pres[i][0])
        tot_acc += accuracy

        print "iteration %d" % (i+1)
        print "num_neg_test_docs: %d" % nres[i][0]
        print "num_neg_training_docs: %d" % nres[i][1]
        print "num_neg_correct_docs: %d" % nres[i][2]
        print "num_pos_test_docs: %d" % pres[i][0]
        print "num_pos_training_docs: %d" % pres[i][1]
        print "num_pos_correct_docs: %d" % pres[i][2]
        print "accuracy: ", "{:.2%}".format(accuracy)
        print "---------------------------------"

    ave_acc = tot_acc / 3.0
    print "ave_accuracy: ", "{:.2%}".format(ave_acc)


def testdocs(nfls, pfls):
    """
    accepts two lists of file names and splits
    them each into 3 randomized subsets. Then for
    each list each subset is tested against the other
    two
    """
    neg_tpl = scramble(nfls)
    pos_tpl = scramble(pfls)

    neg_results = []
    pos_results = []

    # begin the 3-fold testing
    for i in range(3):

        # set empty word dictionary for testing files
        word_dict = {"pos": {}, "neg": {}}


        # change the testing and training files each iteration
        test_neg_fls = neg_tpl[i%3]
        test_pos_fls = pos_tpl[i%3]
        train_neg_fls = neg_tpl[(i+1)%3] + neg_tpl[(i-1)%3]
        train_pos_fls = pos_tpl[(i+1)%3] + pos_tpl[(i-1)%3]
        num_neg = len(train_neg_fls)
        num_pos = len(train_pos_fls)

        # populate the word dictionary with words from
        # reviews tagged with neg
        for f in train_neg_fls:

            neg_wl = getWords(getText(f))
            word_dict = clsWordCounts(word_dict, neg_wl, "neg")

        # add the words from reviews tagged with pos
        for f in train_pos_fls:

            pos_wl = getWords(getText(f))
            word_dict = clsWordCounts(word_dict, pos_wl, "pos")

        # get the number of correctly identified docs
        correct_neg = classify(test_neg_fls, word_dict, "neg", num_neg, num_pos)
        correct_pos = classify(test_pos_fls, word_dict, "pos", num_neg, num_pos)

        # for each iteration add to the two results lists a list containing the
        # number of test files, the number of training files, and the number of
        # correctly identified docs
        neg_results.append([len(test_neg_fls), num_neg, correct_neg])
        pos_results.append([len(test_pos_fls), num_pos, correct_pos])

    # print the results
    get_results(neg_results, pos_results)



# main
def main():

    args = parseArgument()
    directory = args['d'][0]
    fls = getFileNames(directory)
    testdocs(fls[0], fls[1])


# call main
main()


# end naive-bayes
