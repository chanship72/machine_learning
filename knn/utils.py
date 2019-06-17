import numpy as np
from typing import List
from hw1_knn import KNN
from scipy import spatial

def conditional_entropy(branches):
    '''
    branches: C x B array,
              C is the number of branches,
              B is the number of classes
              it stores the number of
              corresponding training samples
              e.g.
                          ○ ○ ○ ○
                          ● ● ● ●
                        ┏━━━━┻━━━━┓
                       ○ ○       ○ ○
                       ● ● ● ●

              branches = [[2,4], [2,0]]
              TOTAL = [6,2]
              entropy = [[2/6,4/6],[2/2,0]]

    '''
    branches = np.array(branches)
    TOTAL = np.sum(branches, axis=1)
    FR = TOTAL / np.sum(TOTAL)
    entropy = branches / TOTAL[:, None]
    entropy = np.array([[-j * np.log2(j) if j > 0 else 0 for j in i] for i in entropy])
    entropy = np.sum(entropy, axis=1) * FR
    entropy = np.sum(entropy)

    return entropy

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float

    return S - conditional_entropy(branches)

def evaluate(node, feature, labels):
    nextNode = node
    while(len(nextNode.children) != 0):
        tmpl = nextNode.dim_split
        tmpc = nextNode.children
        if feature[tmpl] in nextNode.feature_uniq_split:
            idx_child = nextNode.feature_uniq_split.index(feature[tmpl])
            nextNode = tmpc[idx_child]
        else:
            break
    return nextNode.cls_max

def test(node, features, labels):
    correct = 0
    total = 0
    i = 0
    for instance in features:
        total += 1
        label = evaluate(node, instance, labels)
        if label == labels[i]:
            correct += 1
        i+=1
    return float(correct) / float(total)

def mode(labels: List[int]):
    if len(labels) == 0:
        return
    else:
        return max(labels, key=labels.count)

def can_prune(node, features, labels):
    if np.array(features).size == 0 or all(w is None for w in features[0]):
        return True

    correct = 0
    majorLabel = mode(labels if type(labels) is list else labels.tolist())
    for i in range(len(features)):
        if labels[i] == majorLabel:
            correct += 1

    # major > tree
    if float(correct) / len(features) > test(node, features, labels):
        return True

    return False

def traverseTree(treeNode, X_test, y_test):
    if len(treeNode.children) == 0:
        return

    if np.array(X_test).size == 0 or all(w is None for w in X_test[0]):
        treeNode.children = {}
        treeNode.cls_max = mode(y_test if type(y_test) is list else y_test.tolist())
        treeNode.splittable = False
        return


    for S in range(len(treeNode.children)):
        # AW is the list of certain attribute value list
        # AQ is the modified array which converts target attribute as None
        AW = np.array(X_test)[:, treeNode.dim_split]
        AQ = np.array(X_test, dtype=object)
        AQ[:, treeNode.dim_split] = None

        IN = np.where(AW == treeNode.feature_uniq_split[S])
        # YN is the array which includes only certain value corresponding to the target attribute
        YN = np.array(y_test)[IN].tolist()
        #XN is the subset of X_test which only includes certain rows corresponding to target values
        XN = AQ[IN]

        child = treeNode.children[S]
        if child.cls_max in YN:
            traverseTree(child, XN, YN)
        else:
            traverseTree(child, [], [])

    if can_prune(treeNode, X_test, y_test):
        treeNode.children = {}
        treeNode.cls_max = mode(y_test if type(y_test) is list else y_test.tolist())
        treeNode.splittable = False


# TODO: implement reduced error pruning
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    if decisionTree.root_node:
        rootnode = decisionTree.root_node

    traverseTree(rootnode, X_test, y_test)

# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')

#KNN Utils

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    # raise NotImplementedError

    tp, fn, fp, tn = 0, 0, 0, 0
    # print(len(predicted_labels))
    # print(predicted_labels)
    # print(len(real_labels))
    # print(real_labels)
    for idx in range(len(real_labels)):
        if real_labels[idx] == 1 and predicted_labels[idx] == 1:
            tp += 1
        elif real_labels[idx] == 1 and predicted_labels[idx] == 0:
            fn += 1
        elif real_labels[idx] == 0 and predicted_labels[idx] == 1:
            fp += 1
        elif real_labels[idx] == 0 and predicted_labels[idx] == 0:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    recall = tp / (tp + fn) if (tp + fn != 0) else 0

    return ((2 * precision * recall) / (precision + recall)) if (precision + recall != 0) else 0

#TODO: Euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError

    return np.linalg.norm(np.array(point1)-np.array(point2)).tolist()

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError
    return np.inner(point1,point2)

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    # raise NotImplementedError
    dist = 0.0
    for x in range(len(point1)):
        dist += (point1[x] - point2[x])**2
    return -np.exp((-1/2) * dist)

def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    dot = np.dot(point1, point2)
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    cos = dot / (norm1 * norm2)
    return 1 - cos

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        norm = np.linalg.norm(features, axis=1)
        result = list()
        for div in range(len(norm)):
            tmp = (features[div] / norm[div] if norm[div] != 0 else np.array([0]*len(features[div]))).tolist()
            result.append(tmp)
        return result

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        pass
    count = 0

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        self.count += 1
        if self.count <= 1:
            self.feature_fix = features
            self.gmin = np.amin(features, axis=0)
            self.gmax = np.amax(features, axis=0)
            self.gdiff = np.subtract(self.gmax, self.gmin)

        features_scale = np.full((np.array(features).shape[0], np.array(features).shape[1]), np.nan)
        for i in range(np.array(features).shape[0]):
            for j in range(np.array(features).shape[1]):
                if self.gdiff[j] == 0:
                    features_scale[i, j] = 0
                else:
                    features_scale[i, j] = (np.array(features)[i, j] - self.gmin[j]) / self.gdiff[j]

        return features_scale

    # TODO: Complete the model selection function where you need to find the best k
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    best_f1_score, best_k, best_modelname, best_model = -1, 0, None, None
    for name, func in distance_funcs.items():
        best_func_f1_score, best_func_k, best_func_modelname, best_func_model = -1, 0, None, None
        for k in range(1,30,2):
            numK = min(len(Xtrain) - 1, k)
            model = KNN(k=numK, distance_function=func)
            model.train(Xtrain, ytrain)
            train_f1_score = f1_score(ytrain, model.predict(Xtrain))

            valid_f1_score = f1_score(yval, model.predict(Xval))

            # Dont change any print statement
            # print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=numK) +
            #       'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
            #       'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            if valid_f1_score > best_func_f1_score:
                best_func_f1_score, best_func_k, best_func_modelname, best_func_model = valid_f1_score, numK, name, model

        if best_func_f1_score > best_f1_score:
            best_f1_score, best_modelname, best_k, best_model = best_func_f1_score, best_func_modelname, best_func_k, best_func_model

        best_model.train(Xtrain, ytrain)
        # print()
        # print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=best_modelname, best_k=best_k) +
        #       'best f1 score: {best_f1_score:.5f}'.format(best_f1_score=best_f1_score))
        # print()

    return best_model, best_k, best_modelname

# TODO: Complete the model selection function where you need to find the best k with transformation
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    best_f1_score, best_k, best_model, best_modelname, best_scaler = -1, 0, None, None, None
    for scaling_name, scaling_class in scaling_classes.items():
        scaler = scaling_class()
        train_features_scaled = scaler(Xtrain)
        valid_features_scaled = scaler(Xval)

        best_func_f1_score, best_func_k, best_func_model, best_func_modelname, best_func_scaling_name = -1, 0, None, None, None
        for name, func in distance_funcs.items():
            for k in range(1,30,2):
                numK = min(len(Xtrain) - 1, k)
                model = KNN(k=numK, distance_function=func)
                model.train(train_features_scaled, ytrain)
                train_f1_score = f1_score(
                    ytrain, model.predict(train_features_scaled))

                valid_f1_score = f1_score(
                    yval, model.predict(valid_features_scaled))
                # Dont change any print statement
                # print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name,
                #                                                              k=numK) +
                #       'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                #       'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

                if valid_f1_score > best_func_f1_score:
                    best_func_f1_score, best_func_k, best_func_model, best_func_modelname, best_func_scaling_name = valid_f1_score, numK, model, name, scaling_name


            if best_func_f1_score > best_f1_score:
                best_f1_score, best_k, best_model, best_modelname, best_scaler = best_func_f1_score, best_func_k, best_func_model, best_func_modelname, best_func_scaling_name

            # print()
            # print('[part 1.2] {name}\t{scaling_name}\t'.format(name=best_modelname, scaling_name=best_scaler) +
            #       'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k,
            #                                                              test_f1_score=best_f1_score))
            # print()
        if best_func_f1_score > best_f1_score:
            best_f1_score, best_k, best_model, best_modelname, best_scaler = best_func_f1_score, best_func_k, best_func_model, best_func_modelname, best_func_scaling_name

        # print()
        # print('[part 1.2] {name}\t{scaling_name}\t'.format(name=best_modelname, scaling_name=best_scaler) +
        #       'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k,
        #                                                              test_f1_score=best_f1_score))
        # print()
    return best_model, best_k, best_modelname, best_scaler


# mms = MinMaxScaler()
# resultA = mms([[2, -1], [-1, 5], [0, 0]])
# print(resultA)
# ns = NormalizationScaler()
# resultA = ns([[3, 4], [1, -1], [0, 0]])
# print(resultA)


# from data import data_processing
# Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
# distance_funcs = {
#     'euclidean': euclidean_distance,
#     'gaussian': gaussian_kernel_distance,
#     'inner_prod': inner_product_distance,
#     'cosine_dist': cosine_sim_distance,
# }
# best_model, best_k, best_function = model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval)

# IG = Information_Gain(0.97,[[5,2],[3,10]])
# print(IG)