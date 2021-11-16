
from hw2_utils import epoch_loss
import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt





def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''

    N,d = x_train.shape
    a = torch.zeros(N,requires_grad=True)
    ones = torch.ones(N,1)
    # print(a,ones)
    t = 0
    A = torch.zeros(N,N)
    for i in range(N):
        for j in range(N):
            A[i][j] = y_train[i]*y_train[j]*kernel(x_train[i], x_train[j])
    # print(A)
    # print("Size of A is:", A.shape)
    while t < num_iters:
        # print("iteration ",t )
        t+=1
        # print("Size of a is:",a.shape)
        # print("Size of matmul((A+A.t()),a) is:",torch.matmul((A+A.t()),a).shape)
        h =-torch.matmul(ones.t(),a.T)+ 1/2*torch.matmul(torch.matmul(a,A),a.T)
        #h = -1/2*torch.matmul(torch.matmul(a,A),a.t())+torch.matmul(ones.t(),a.t())
        #h = torch.matmul(torch.ones(N).T,a)-0.5*torch.matmul(torch.matmul(a.T,A),a)
        if a.grad:    
            a.grad.zero_()
        # print(a.requires_grad)
        
        h.backward()
        
        # g = -1/2*torch.matmul((A+A.t()),a)+ones
        # print("Size of g is:",g.shape)
        with torch.no_grad():
            a =torch.clamp_(a -lr*a.grad,0,c )
            
        a.requires_grad_()
        


    # optimizer = torch.optim.SGD([a],lr)
    # for i in range(num_iters):
    #     h = -1/2*torch.matmul(torch.matmul(a,A),a.t())+torch.matmul(ones.t(),a.t())
    #     print(h)
    #     optimizer.zero_grad()
    #     h.backward()
    #     optimizer.step()

    # print("Size of a is:",a.shape)
    return a



def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    N,d = x_train.shape
    M,_ = x_test.shape
    #print(x_test.shape)
    y_test = torch.zeros(M)
    for m in range(M):
        w = 0
        for i in range(N):
            w+=alpha[i]*y_train[i]*kernel(x_train[i],x_test[m])
        y_test[m] = w
    #print(w.shape)

    return y_test

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        self.conv1=nn.Conv2d(1,8,kernel_size=3,stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8,4,kernel_size=3,stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.fc = nn.Linear(4,10,bias=True)
        self.relu = nn.ReLU()
        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        #print("Forwarding")
        N,_,_ = xb.shape
        xb = xb.view(N,1,8,8)
        
        y = self.relu(self.conv1(xb))   # N, 8, 6, 6

        #print("Shape of Conv1",y.shape)
        y = self.maxpool(y)  # N, 8, 3, 3   
        #print("Shape of maxpool",y.shape)
        y = self.relu(self.conv2(y))    # N, 4, 1, 1
        #print("Shape of Conv2",y.shape)
        y = y.view(N,4)
        y = self.fc(y)      # N, 1,1, 10
        #print("Shape of fc",y.shape)
        y = y.view(N,10)
        # y = self.relu(y)
        return y

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs,scheduler=None, batch_size=1, ):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)
    with torch.no_grad():
        train_losses.append(epoch_loss(net, loss_func, train_dl ))
        test_losses.append(epoch_loss(net, loss_func, test_dl ))
    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        for i, (xb, yb) in enumerate(train_dl):

            train_loss = loss_func(net(xb), yb)
            # print("Doing Backward")
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(epoch_loss(net, loss_func, train_dl ))
        test_losses.append(epoch_loss(net, loss_func, test_dl ))
        if scheduler!=None:
            scheduler.step()

    return train_losses, test_losses


