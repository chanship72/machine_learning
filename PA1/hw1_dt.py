import numpy as np
from typing import List
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

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

        uniq_attr = [0]
        cls_size = np.max(self.labels) + 1
        BRANCH = np.zeros((len(uniq_attr), cls_size))
        for i, val in enumerate(uniq_attr):
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

    def split(self):
        if not 'maxv' in locals():
            maxv = -1
        for idx_dim in range(len(self.features[0])):
            feature_attr = np.array(self.features)[:, idx_dim]
            if None in feature_attr:
                continue
            uniq_feature_attr = np.unique(feature_attr)
            cls_size = np.max(self.labels) + 1
            branch = np.zeros((len(uniq_feature_attr), cls_size))
            for i, val in enumerate(uniq_feature_attr):
                y = np.array(self.labels)[np.where(feature_attr == val)]
                for yi in y:
                    branch[i, yi] = branch[i, yi] + 1
            IG = Util.Information_Gain(self.entropy,branch)
            # print(str(idx_dim) + "IG:"+str(IG)+"/maxv:"+str(maxv))
            if IG > maxv or idx_dim == 0:
                maxv = IG
                self.dim_split = idx_dim
                self.feature_uniq_split = uniq_feature_attr.tolist()
            elif IG == maxv:
                if len(uniq_feature_attr.tolist()) > len(self.feature_uniq_split):
                    self.dim_split = idx_dim
                    self.feature_uniq_split = uniq_feature_attr.tolist()

        target_attr_column = np.array(self.features)[:, self.dim_split]
        filtered_feature_list = np.array(self.features, dtype=object)
        filtered_feature_list[:, self.dim_split] = None

        for splitTarget in self.feature_uniq_split:
            idx_matched_samples = np.where(target_attr_column == splitTarget)
            matched_label_list = np.array(self.labels)[idx_matched_samples].tolist()
            matched_sample_list = filtered_feature_list[idx_matched_samples].tolist()

            child = TreeNode(matched_sample_list, matched_label_list, self.num_cls)

            if np.array(matched_sample_list).size == 0 or all(w is None for w in matched_sample_list[0]):
                child.splittable = False

            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature):
        if self.splittable and (feature[self.dim_split] in self.feature_uniq_split):
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



