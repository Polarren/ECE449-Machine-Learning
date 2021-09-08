import torch
import hw1_utils as utils


'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else (otherwise, your w will be ordered differently than the
    reference solution's in the autograder)!!!
'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    N,d = torch.tensor.size(X)
    w = torch.zeros(d+1,1)
    ones = torch.ones(N,1)
    X = torch.cat((ones,X),1)
    t = 0
    while t<num_iter:
        t+=1
        g = 1/N * (torch.matmul(X.t(),torch.matmul(X,w)-Y))
        w = w -lrate*g

    return w



def linear_normal(X, Y):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    pass


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    pass


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides) 
    '''
    pass


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass
