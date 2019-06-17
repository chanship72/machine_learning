import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')
    # print(x)
    centers = [generator.randint(0,n)]

    #print("center:" + str(centers))
    for i in range(1,n_cluster):
        D2 = np.array([np.min([np.linalg.norm(xi - x[c])**2 for c in centers]) for xi in x])
        probs = D2/np.sum(D2)
        i = np.argmax(probs, axis=0)
        centers.append(i)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

#print(get_k_means_plus_plus_center_indices(5, 3, np.array([[0,1],[1,2],[2,3],[3,4],[4,5]]), generator=np.random))

def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def distance(self, vec1, vec2):
        sub = np.subtract(vec1, vec2)
        return np.inner(sub, sub)

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeans class')

        # centroids = np.zeros((self.n_cluster, D))
        def dist(centroid, x, y):
            N = x.shape[0]
            return np.sum([np.sum((x[y == k] - centroid[k]) ** 2) for k in range(self.n_cluster)]) / N

        # Get random means
        centroids = x[self.centers]

        y = np.zeros(N, dtype=int)
        J = dist(centroids, x, y)

        iter = 0
        while iter < self.max_iter:
            l2 = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)
            y = np.argmin(l2, axis=0)
            J_new = dist(centroids, x, y)
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new

            centroids = np.array([np.mean(x[y == k], axis=0) for k in range(self.n_cluster)])
            iter += 1

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, iter


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeansClassifier class')

        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x, centroid_func)
        V = [{} for k in range(self.n_cluster)]
        for y_i, r_i in zip(y, membership):
            if y_i not in V[r_i].keys():
                V[r_i][y_i] = 1
            else:
                V[r_i][y_i] += 1
        centroid_labels = np.array([max(V_k, key=V_k.get) if V_k is not None else 0 for V_k in V])

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def distance(self,vec1,vec2):
        sub = np.subtract(vec1,vec2)
        return np.inner(sub,sub)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')
        distance = self.distance
        labels = []
        for m in range(N):
            dist_obs = [distance(x[m,:],self.centroids[n,:]) for n in range(self.n_cluster)]
            labels.append(self.centroid_labels[np.argmin(dist_obs)])
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #          'Implement transform_image function')

    cluster = code_vectors.shape[0]

    def distance(p1, p2):
        diff = np.subtract(p1, p2)
        return np.inner(diff,diff)

    new_im = np.zeros((image.shape[0],image.shape[1],3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dfrom = [distance(image[i,j,:],code_vectors[k,:]) for k in range(cluster)]
            new_im[i,j,:] = code_vectors[np.argmin(dfrom),:]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

