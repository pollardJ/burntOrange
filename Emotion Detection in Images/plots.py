# Authors: David Wen, Alex Romriell, Jacob Pollard

from glob import glob
import numpy as np
import seaborn
import matplotlib.pyplot as plt

# STYLES = plt.style.available

CSVDIR = "/Users/jtpollard/MSAN/msan630/Emotobot/csvfiles/"
PLOTSDIR = "/Users/jtpollard/MSAN/msan630/Emotobot/plots/"


def get_csvfiles(csvdir):
    """
    -get the csv files from the passed directory
    :param csvdir:
    :return:
    """
    csv_filenames = glob('{}*.csv'.format(csvdir))

    return csv_filenames


def get_array_from_csv(csvfile):
    """
    -read in a csv file containing model diagnostics and return a numpy array
    and the name components for using in the plot titles
    :param csvfile:
    :return:
    """
    model_data = np.genfromtxt(csvfile, delimiter=",", skip_header=1)
    plot_title = csvfile.split("/")[-1].split(".")[0].split("_")[3:5]

    return model_data, plot_title



if __name__ == "__main__":

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'unhappy']

    color_map = dict()
    color_map['angry'] = 'red'
    color_map['disgust'] = 'green'
    color_map['fear'] = 'purple'
    color_map['happy'] = 'y'
    color_map['neutral'] = 'black'
    color_map['surprise'] = 'pink'
    color_map['unhappy'] = 'blue'

    # dict for the name of the expression to show on the graph
    graph_emos = dict()
    graph_emos['angry'] = 'Anger'
    graph_emos['disgust'] = 'Disgust'
    graph_emos['fear'] = 'Fear'
    graph_emos['happy'] = 'Happiness'
    graph_emos['neutral'] = 'Neutrality'
    graph_emos['surprise'] = 'Surprise'
    graph_emos['unhappy'] = 'Sadness'


    csv_files = get_csvfiles(CSVDIR)
    # print csv_files

    model_data = list()
    title_emos = list()
    for c in csv_files:
        df, tl = get_array_from_csv(c)
        model_data.append(df)
        title_emos.append(tl)

    n = len(model_data)

    for i in range(n):

        plt.clf()
        plt.style.use('seaborn-notebook')
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.xlabel('Number of Epochs', fontsize=15)
        plt.ylabel('Cross Entropy Loss', fontsize=15)
        # plt.title('Cross Entropy Loss for Model Classifying\n'
        #           'Faces Expressing {} vs {}'.format(graph_emos[title_emos[i][0]], graph_emos[title_emos[i][1]]))
        epochs = model_data[i][:, 0]
        train_loss = model_data[i][:, 1]
        test_loss = model_data[i][:, 2]
        plt.plot(epochs, train_loss, color="blue", label="Train")
        plt.plot(epochs, test_loss, color="darkorange", label="Test")
        plt.legend(loc="best", prop={'size': 20})
        savepath = PLOTSDIR + "val_curve_{}_{}.png".format(title_emos[i][0], title_emos[i][1])
        plt.savefig(savepath, dpi=500, bbox_inches="tight")
        # plt.show()

