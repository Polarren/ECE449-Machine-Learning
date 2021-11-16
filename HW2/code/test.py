import hw2_utils
import hw2
import hw1_utils 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

def pdf (x_test):
    X, Y = hw1_utils.load_logistic_data()
    a = hw2.svm_solver(X,Y,0.01, 5, )
    print(a)
    Y_test = hw2.svm_predictor(a,X,Y,x_test)
    return Y_test


def test_1():
    X, Y = hw1_utils.load_logistic_data()
    a = hw2.svm_solver(X,Y,0.01, 11,hw2_utils.poly(degree=2))
    Y_test = hw2.svm_predictor(a,X,Y,X)
    
    N,d = X.shape
    y = []

    for i in range(N):
        if Y_test[i]> 0:
            y.append("black")
        else: 
            y.append("green")
    y = np.array(y)
    plt.figure()
    plt.scatter(X.t()[0],X.t()[1],c=y)
    plt.show()

    return 1


def test_2():
    net = hw2.DigitsConvNet()
    train, test = hw2_utils.torch_digits()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    train_losses, test_losses=hw2.fit_and_evaluate( net, optimizer, criterion, train, test, 30, batch_size=1)
    print ("train losses: ", train_losses)
    print("test losses: ", test_losses)


def Q2_d():
    x_train,y_train =hw2_utils.xor_data()
    alpha = hw2.svm_solver(x_train,y_train,0.1, 10000,hw2_utils.poly(degree=2))
    plt.subplot(221)
    hw2_utils.svm_contour(pred_fxn1)
    return

def pred_fxn1(x_test) :
    return hw2.svm_predictor(alpha1, x_train, y_train, x_test,kernel=hw2_utils.poly(degree=2))

def pred_fxn2(x_test) :
    return hw2.svm_predictor(alpha2, x_train, y_train, x_test,kernel=hw2_utils.rbf(1))

def pred_fxn3(x_test) :
    return hw2.svm_predictor(alpha3, x_train, y_train, x_test,kernel=hw2_utils.rbf(2))

def pred_fxn4(x_test) :
    return hw2.svm_predictor(alpha4, x_train, y_train, x_test,kernel=hw2_utils.rbf(4))

def Q4_c():
    net = hw2.DigitsConvNet()
    train, test = hw2_utils.torch_digits()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    train_losses, test_losses=hw2.fit_and_evaluate( net, optimizer, criterion, train, test, 30, batch_size=1)
    # print ("train losses: ", train_losses)
    # print("test losses: ", test_losses)
    return train_losses, test_losses

def Q4_d():
    net = hw2.DigitsConvNet()
    train, test = hw2_utils.torch_digits()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    train_losses, test_losses=hw2.fit_and_evaluate( net, optimizer, criterion, train, test, 30,scheduler, batch_size=1)
    # print ("train losses: ", train_losses)
    # print("test losses: ", test_losses)
    return train_losses, test_losses

def Q4_e():
    net = hw2.DigitsConvNet()
    train, test = hw2_utils.torch_digits()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    train_losses, test_losses=hw2.fit_and_evaluate( net, optimizer, criterion, train, test, 30, batch_size=16)
    # print ("train losses: ", train_losses)
    # print("test losses: ", test_losses)
    return train_losses, test_losses

def plot_cde():
    train_losses_c, test_losses_c = Q4_c()
    train_losses_d, test_losses_d = Q4_d()
    train_losses_e, test_losses_e = Q4_e()
    plt.figure()
    epochs_c = np.arange(len(train_losses_c))
    epochs_d = np.arange(len(train_losses_d))
    epochs_e = np.arange(len(train_losses_e))
    plt.plot(epochs_c,train_losses_c,label='train batch = 1')
    plt.plot(epochs_c,test_losses_c,label='test batch = 1')
    plt.plot(epochs_d,train_losses_d,label='train decayed lr = 1')
    plt.plot(epochs_d,test_losses_d,label='test decayed lr = 1')
    plt.plot(epochs_e,train_losses_e,label='train batch = 16')
    plt.plot(epochs_e,test_losses_e,label='test batch = 16')
    plt.legend()
    plt.title('HW2_4(e)')
    plt.xlabel('epochs')
    plt.ylabel('training')
    plt.show()
# test_2()


# plot_cde()
plt.subplots()

x_train,y_train =hw2_utils.xor_data()
alpha1 = hw2.svm_solver(x_train,y_train,0.1, 10000,hw2_utils.poly(degree=2))
plt.subplot(221,title = 'poly kernel')
hw2_utils.svm_contour(pred_fxn1)
x_train,y_train =hw2_utils.xor_data()
alpha2 = hw2.svm_solver(x_train,y_train,0.1, 10000,hw2_utils.rbf(1))
plt.subplot(222,title = 'RBF with $\sigma$=1')
hw2_utils.svm_contour(pred_fxn2)
x_train,y_train =hw2_utils.xor_data()
alpha3 = hw2.svm_solver(x_train,y_train,0.1, 10000,hw2_utils.rbf(2))
plt.subplot(223,title = 'RBF with $\sigma$=2')
hw2_utils.svm_contour(pred_fxn3)
x_train,y_train =hw2_utils.xor_data()
alpha4 = hw2.svm_solver(x_train,y_train,0.1, 10000,hw2_utils.rbf(4))
plt.subplot(224,title = 'RBF with $\sigma$=4')
hw2_utils.svm_contour(pred_fxn4)
plt.tight_layout()
plt.show()