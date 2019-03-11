from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

class KNN:
    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        #raise NotImplementedError
        self.features_train = features
        self.labels_train = labels

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError
        label_predict = [None]*len(features)
        for index, item in enumerate(features):
            # print("item:",item)
            kneighbor_list = self.get_k_neighbors(item)
            vote = [0] * 2
            for label_idx in kneighbor_list:
                if label_idx == 0:
                    vote[0] += 1
                elif label_idx == 1:
                    vote[1] += 1
            if vote[0] >= vote[1]:
                label_predict[index] = 0
            else:
                label_predict[index] = 1

        # print("@KNN predicted:",label_predict)
        return label_predict

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        distance_list = list()

        for index_train, item_train in enumerate(self.features_train):
            cur_dist = self.distance_function(point, item_train)
            distance_list.append((cur_dist, index_train))

        distance_list.sort(key=lambda x: x[0])
        # print(distance_list)

        kneighbors = []

        numK = min(len(distance_list)-1, self.k)
        for i in range(numK):
            kneighbors.append(self.labels_train[distance_list[i][1]])

        return kneighbors


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
