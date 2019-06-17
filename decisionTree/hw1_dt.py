import numpy as np
from typing import List
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    # TODO: train Decision Tree
    def train(self, features, labels):
        # raise NotImplementedError
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        UNIQ_ATTR = [0]
        cls_size = np.max(self.labels) + 1
        BRANCH = np.zeros((len(UNIQ_ATTR), cls_size))
        for i, val in enumerate(UNIQ_ATTR):
            y = np.array(self.labels)
            for yi in y:
                BRANCH[i, yi] = BRANCH[i, yi] + 1

        self.entropy = Util.conditional_entropy(BRANCH)
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        if not 'ME' in locals():
            ME = -1
        for idx_dim in range(len(self.features[0])):
            ATTR = np.array(self.features)[:, idx_dim]
            if None in ATTR:
                continue
            UNIQ_ATTR = np.unique(ATTR)
            cls_size = np.max(self.labels) + 1
            BRANCH = np.zeros((len(UNIQ_ATTR), cls_size))
            for i, val in enumerate(UNIQ_ATTR):
                y = np.array(self.labels)[np.where(ATTR == val)]
                for yi in y:
                    BRANCH[i, yi] = BRANCH[i, yi] + 1
            IG = Util.Information_Gain(self.entropy,BRANCH)
            # print(str(idx_dim) + "IG:"+str(IG)+"/ME:"+str(ME))
            if IG > ME or idx_dim == 0:
                ME = IG
                self.dim_split = idx_dim
                self.feature_uniq_split = UNIQ_ATTR.tolist()
            elif IG == ME:
                if len(UNIQ_ATTR.tolist()) > len(self.feature_uniq_split):
                    self.dim_split = idx_dim
                    self.feature_uniq_split = UNIQ_ATTR.tolist()

        AW = np.array(self.features)[:, self.dim_split]
        AQ = np.array(self.features, dtype=object)
        AQ[:, self.dim_split] = None

        for S in self.feature_uniq_split:

            IN = np.where(AW == S)

            YN = np.array(self.labels)[IN].tolist()

            XN = AQ[IN].tolist()

            child = TreeNode(XN, YN, self.num_cls)

            if np.array(XN).size == 0 or all(w is None for w in XN[0]):
                child.splittable = False

            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    # TODO:treeNode predict function
    def predict(self, feature):
        if self.splittable and (feature[self.dim_split] in self.feature_uniq_split):
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max

#
# import data
# import hw1_dt as decision_tree
# import utils as Utils
# from sklearn.metrics import accuracy_score
#
# #load data
# X_train, y_train, X_test, y_test = data.load_decision_tree_data()
#
# # set classifier
# dTree = decision_tree.DecisionTree()
#
# # # training
# dTree.train(X_train.tolist(), y_train.tolist())
# Utils.print_tree(dTree)
#
# Utils.reduced_error_prunning(dTree, X_test, y_test)
#
# Utils.print_tree(dTree)
#
#
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)


