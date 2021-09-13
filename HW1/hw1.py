import torch
import hw1_utils as utils
import matplotlib.pyplot as plt
import numpy

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
    N,d = X.shape
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
    N,d = X.shape
    ones = torch.ones(N,1)
    X = torch.cat((ones,X),1)
    w = torch.matmul( torch.pinverse(X), Y)
    return w



def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''

    X,Y=utils.load_reg_data()
    N,d = X.shape
    
    w = linear_normal(X,Y)
    ones = torch.ones(N,1)
    # X = torch.cat((ones,X),1)
    
    # print("Dimension of X is: "+str(X.shape))
    plot = plt.figure()
    plt.scatter(X,Y)
    plt.plot(X,torch.matmul(torch.cat((ones,X),1),w))
    plt.title('HW1_2(c):Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig('HW1_2(c).png')
    return plot


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
    N,d = X.shape
    w = torch.zeros(d+1,1)
    ones = torch.ones(N,1)
    X = torch.cat((ones,X),1)
    t = 0
    # print("Dimension of X is: "+str(X.shape))
    # print("Dimension of X is: "+str(X.shape))
    # print("Dimension of w is: "+str(w.shape))
    while t < num_iter:
        sum = torch.zeros(d+1,1)
        for i in range(N):
            exp = torch.exp(-Y[i]*torch.matmul(X[i],w))
            # print ('sum=',sum)
            # print ('newsum =', (-Y[i] * X[i].reshape((d+1,1))*exp)/(1+exp))
            sum += (-Y[i] * X[i].reshape((d+1,1))*exp)/(1+exp)
        grad = sum/N
        # if t<3:
        #     print("grad is: "+str(grad))
        #     print('w is :',w)
        w = w - lrate*grad
        t+=1
    # print("Dimension of exp is: "+str(exp.shape))
    # print("Dimension of w is: "+str(w.shape))
    return w



def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X,Y=utils.load_logistic_data()
    N,d = X.shape
    #print('N,d is : ',N,d)
    w_lin = linear_gd(X,Y)
    w_logi = logistic(X,Y)
    ones = torch.ones(N,1)
    # X = torch.cat((ones,X),1)
    
    # print("Dimension of X is: "+str(X.shape))
    plot = plt.figure()
    plt.scatter(X.t()[0],X.t()[1])
    x2_lin = - w_lin[1]/w_lin[2]*X.t()[0]-w_lin[0]/w_lin[2]
    x2_logi = - w_logi[1]/w_logi[2]*X.t()[0]-w_logi[0]/w_logi[2]
    plt.plot(X.t()[0],x2_lin,label='Linear')
    plt.plot(X.t()[0],x2_logi,label='Logistic')
    plt.legend()
    plt.title('HW1_3(c)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # plt.savefig('HW1_2(c).png')
    return plot
