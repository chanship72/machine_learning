"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    err = np.sum((y - np.dot(X,w))**2) / float(len(y))
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  w = (X^T * X)^-1 X^T y
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, D)
    - regularizer : (D, num_samples)
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################

    eigenvalue = np.linalg.eigvals(np.dot(X.T,X))

    finvert = True
    for v in eigenvalue:
        if v < 10**-5:
            finvert = False

    if not finvert:
        regularizer = 0.1 * (np.identity(len(X[0])))
    else:
        regularizer = 0.0
    w = np.dot(np.linalg.inv(np.dot(X.T,X) + regularizer), np.dot(X.T,y))
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    regularizer = lambd * (np.identity(len(X[0])))
    w = np.dot(np.linalg.inv(np.dot(X.T,X) + regularizer), np.dot(X.T,y))
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    bestlambda = None
    bestErr = float('inf')

    for lambd in range(-19,19,1):
        w = regularized_linear_regression(Xtrain, ytrain, np.float_power(10,lambd))
        err = mean_square_error(w, Xval, yval)
        # print("w:"+str(w))
        # print("err:"+str(err))

        if err < bestErr:
            bestErr = err
            bestlambda = 10**lambd

    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    p = 2
    # print("X.shape:" + str(X.shape))

    concate = X
    while p <= power:
        temp = np.power(X,p)
        concate = np.concatenate((concate, temp), axis=1)
        p = p+1
    # matrix = list()
    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         X[i][j] = X[i][j]**(j+1)
    #print(concate)
    # print("X.shape:" + str(concate.shape))
    return concate


