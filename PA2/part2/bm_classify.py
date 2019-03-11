import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    y = np.array(list(map(lambda y: -1 if y==0 else y, y)))

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        eX = np.append(np.ones((N,1)),X, axis=1)
        ew = np.append(b, w)

        def indicatorF(arr):
            return np.array(list(map(lambda x: 1 if x <= 0.0 else 0, arr)))

        for i in range(max_iterations):
            # indicator : (N,1)
            indicator = indicatorF(np.dot(eX,ew)*y)
            n = np.count_nonzero(indicator)
            gd = np.dot(eX.T,indicator*y)/N
            ew = ew + step_size * gd

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        eX = np.append(np.ones((N,1)),X, axis=1)
        ew = np.append(b,w)

        for i in range(max_iterations):
            # indicator : (N,1)
            sigmoidTerm = sigmoid(-np.dot(eX, ew)*y)
            dF_w_b = np.dot(eX.T, sigmoidTerm*y)/N
            ew = ew + step_size * dF_w_b
    else:
        raise "Loss Function is undefined."

    w = ew[1:]
    b = ew[0]
    #print(w,b)
    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    ############################################
    return 1 / (1 + np.exp(-value))

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        preds = np.array(list(map(lambda x, y: 1.0 if y>x else 0.0,preds,np.dot(X,w) + b)))

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        preds = np.array(list(map(lambda x: 1.0 if x>=0.5 else 0.0,sigmoid(np.dot(X,w) + b))))

    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    Ycls = np.zeros((N,C))  # NxC
    for i in range(N):
        Ycls[i][y[i]] = 1

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        # eX = np.append(np.ones((N, 1)), X, axis=1)    # NxD+1
        # b = np.reshape(b, (C, 1))
        # ew = np.append(b, w, axis=1)    # CxD+1

        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x))

        for k in range(max_iterations):
            random_idx = np.random.choice(N,1)
            X_i = X[random_idx]
            Ycls_i = Ycls[random_idx]
            exp = np.dot(w,X_i.T) + b.reshape(C,1)   # CxD x Dx1  = Cx1
            exp = exp - np.max(exp)
            softX = soft_max(exp)
            softX = softX - Ycls_i.T  # Cx1 - Cx1
            dw = np.dot(softX,X_i)   # Cx1 x 1xD
            db = softX.flatten()   # C x None

            w = w - step_size * dw
            b = b - step_size * db

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        eX = np.append(np.ones((N,1)),X, axis=1)    # NxD+1
        b = np.reshape(b, (C, 1))
        ew = np.append(b, w, axis=1)    # CxD+1

        for k in range(max_iterations):
            grad = np.zeros(C)
            exp = np.exp(np.dot(eX,ew.T)) / np.sum(np.exp(np.dot(eX,ew.T)),axis=1)[:,None]
            exp = exp - Ycls
            grad = np.dot(exp.T,eX) / N
            ew = ew - step_size * grad

        #print("ew.shape:" + str(ew.shape))
        w = ew[:,1:]
        b = ew[:,0]

    else:
        raise "Type of Gradient Descent is undefined."

    #print(w,b)
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################

    C = w.shape[0]

    for i in range(N):
        cls = np.zeros(C)
        for k in range(C):
            cls[k] = b[k] + np.dot(w[k], X[i].T)
        preds[i] = np.argmax(cls)

    assert preds.shape == (N,)
    return preds

