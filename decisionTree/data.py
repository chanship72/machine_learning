import numpy as np
import pandas as pd


# ''' All data utilities are here'''

def sample_decision_tree_data():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
    labels = [0, 0, 1, 1]
    return features, labels


def sample_decision_tree_test():
    features = [['a', 'b'], ['b', 'a'], ['b', 'b']]
    labels = [0, 0, 0]
    return features, labels


def load_decision_tree_data():
    f = open('car.data', 'r')
    white = [[int(num) for num in line.split(',')] for line in f]
    white = np.asarray(white)

    [N, d] = white.shape

    ntr = int(np.round(N * 0.66))
    ntest = N - ntr

    Xtrain = white[:ntr].T[:-1].T
    ytrain = white[:ntr].T[-1].T
    Xtest = white[-ntest:].T[:-1].T
    ytest = white[-ntest:].T[-1].T

    return Xtrain, ytrain, Xtest, ytest


def data_processing():
    white = pd.read_csv('heart_disease.csv', low_memory=False, sep=',', na_values='?').values

    [N, d] = white.shape

    np.random.shuffle(white)
    # prepare data

    ntr = int(np.round(N * 0.8))
    nval = int(np.round(N * 0.15))
    ntest = N - ntr - nval
    # spliting training, validation, and test
    Xtrain = np.append([np.ones(ntr)], white[:ntr].T[:-1], axis=0).T
    ytrain = white[:ntr].T[-1].T
    Xval = np.append([np.ones(nval)], white[ntr:ntr + nval].T[:-1], axis=0).T
    yval = white[ntr:ntr + nval].T[-1].T
    Xtest = np.append([np.ones(ntest)], white[-ntest:].T[:-1], axis=0).T
    ytest = white[-ntest:].T[-1].T
    # print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)
    return Xtrain, ytrain, Xval, yval, Xtest, ytest


def most_common(lst):
    return max(set(lst), key=lst.count)

