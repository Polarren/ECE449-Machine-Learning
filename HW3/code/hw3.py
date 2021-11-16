import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import _MaskedUnaryOperation

from hw3_utils import visualization, get_dataset_fixed


class Stump():
    def __init__(self, data, labels, weights):
        '''
        Initializes a stump (one-level decision tree) which minimizes
        a weighted error function of the input dataset.

        In this function, you will need to learn a stump using the weighted
        datapoints. Each datapoint has 2 features, whose values are bounded in
        [-1.0, 1.0]. Each datapoint has a label in {+1, -1}, and its importance
        is weighted by a positive value.

        The stump will choose one of the features, and pick the best threshold
        in that dimension, so that the weighted error is minimized.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
            labels: An ndarray with shape (n, ). Values are +1 or -1.
            weights: An ndarray with shape (n, ). The weights of each
                datapoint, all positive.
        '''
        # You may choose to use the following variables as a start

        # The feature dimension which the stump will decide on
        # Either 0 or 1, since the datapoints are 2D
        self.dimension = 0

        # The threshold in that dimension
        # May be midpoints between datapoints or the boundaries -1.0, 1.0
        self.threshold = -1.0

        # The predicted sign when the datapoint's feature in that dimension
        # is greater than the threshold
        # Either +1 or -1
        self.sign = 1
        
        n,dim = data.shape # dim = 2 
        if weights is None:
            weights = [1.0] * labels.size
        # Create thresold lists
        threshold_list= np.zeros((dim, n+1))
        for d in range(dim):
            data_sorted = np.copy(data)
            data_sorted = data_sorted.transpose()
            data_sorted[d] = np.sort(data_sorted[d])
            #print(data)
            threshold_list[d][0]=-1
            threshold_list[d][-1]=1
            for i in range(n-1):
                threshold_list[d][i+1] = (data_sorted[d][i]+data_sorted[d][i+1])/2

        #print("threshold list is",threshold_list)
        #print(data)
        # Calculate loss
        L = np.zeros((dim,n+1))
        Y_hat = np.zeros((n,1))
        s_list = np.zeros((dim,n+1))
        for d in range(dim):
            for i in range(n+1):
                # Construct Y_hat
                L_max = np.inf 
                for s in [-1,1]:
                    
                    for j in range(n):
                        if data[j][d]>=threshold_list[d][i]:
                            Y_hat[j][0] = s
                        else:
                            Y_hat[j][0] = -s
                    # print(Y_hat[0][0],labels[0] )
                    one_function = np.array([1 if Y_hat[j][0]!=labels[j] else 0 for j in range(n) ])
                    loss = np.matmul(np.array(weights).T,one_function)
                    if loss< L_max:
                        #print(d,i)
                        L[d][i] = loss
                        L_max  = loss
                        s_list[d][i]= s 
                        
        # print("Loss is ", L)
        min_index = np.argmin(L)
        
        self.dimension = int(min_index/(n+1))
        self.threshold = np.reshape(threshold_list,-1)[min_index]
        self.sign =  np.reshape(s_list,-1)[min_index]

        # print ("min_index = ", min_index)
        # print("Thresold = ", self.threshold)
        # print("dimension = ", self.dimension)
        # print("sign = ", self.sign)

        pass

    def predict(self, data):
        '''
        Predicts labels of given datapoints.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].

        Returns:
            prediction: An ndarray with shape (n, ). Values are +1 or -1.
        '''
        n,_ = data.shape # dim = 2 
        Y = np.array([self.sign if data[i][self.dimension]>=self.threshold else -self.sign for i in range(n)])
        return Y



def bagging(data, labels, n_classifiers, n_samples, seed=0):
    '''
    Runs Bagging algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.
        n_samples: Number of samples to train each classifier.
        seed: Random seed for NumPy.

    Returns:
        classifiers: A list of classifiers.
    '''
    classifiers = []
    n = data.shape[0]

    for i in range(n_classifiers):
        np.random.seed(seed + i)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        data_sample = np.array([data[i] for i in sample_indices])
        label_sample = np.array([labels[i] for i in sample_indices])
        stump = Stump(data_sample,label_sample,None)
        classifiers.append(stump)
        

    return classifiers


def adaboost(data, labels, n_classifiers):
    '''
    Runs AdaBoost algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.

    Returns:
        classifiers: A list of classifiers.
        weights: A list of weights assigned to the classifiers.
    '''
    classifiers = []
    weights = []
    n = data.shape[0]
    data_weights = np.ones(n) / n

    for i in range(n_classifiers):
        stump = Stump(data, labels, data_weights)
        classifiers.append(stump)
        Y_hat = stump.predict(data)
        epsilon = np.matmul(data_weights,(Y_hat==labels).T)
        #print("epsilon = ",epsilon)
        alpha = -1/2*np.log((1-epsilon)/epsilon)
        #print("alpha = ",alpha)
        weights.append(alpha)

        Zt= np.matmul(data_weights,np.exp(-alpha*labels*Y_hat))
        # Update data weight
        data_weights = data_weights * np.exp(-alpha*labels*Y_hat)/Zt
        # for i in range(len(data_weights)):
            
        #     data_weights[i]=data_weights[i]*np.exp(-alpha*labels[i]*Y_hat[i])/Zt


    return classifiers, weights


if __name__ == '__main__':
    data, labels = get_dataset_fixed()

    # You can play with the dataset and your algorithms here
    # classifier = Stump()
