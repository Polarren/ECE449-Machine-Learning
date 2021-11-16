import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=3):
    """K-Means.
    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a cluster center.
    
    Return:
        c: shape [2, 2]. Each column is a cluster center.
    """

    # loading data and intiailzation of the cluster centers
    if X is None:
        X, c = hw4_utils.load_data()
    else:
        c = init_c
    # your code below

    _,N=X.shape
    
    for k in range(n_iters):
        # first solve the assignment problem given the centers c
        classes = torch.zeros(N)
        c_0 = torch.zeros(2)
        c_1 = torch.zeros(2)
        count_0 = 0
        count_1 = 0
        for i in range(N):
            classes[i] = classify([X[0][i],X[1][i]],c)
        # then solve the cluster center problem given the assignments
        for i in range(N):
            if classes[i]==0: 
                c_0[0] += X[0][i]
                c_0[1] += X[1][i]
                count_0 +=1
            else:
                c_1[0]  += X[0][i]
                c_1[1]  += X[1][i]
                count_1 +=1
        c_0 = c_0/count_0
        c_1 = c_1/count_1

        x1  = X[:,[i for i in range(N) if classes[i]==0]]
        x2  = X[:,[i for i in range(N) if classes[i]==1]]
        print("c1:",c_0)
        print("c2:",c_1)
        c = torch.cat((c_0.reshape((2,1)), c_1.reshape((2,1))), dim=1)
        print(c)
        # visulize the current clustering using hw4_utils.vis_cluster. 
        hw4_utils.vis_cluster(c_0.reshape((2,1)),x1 , c_1.reshape((2,1)), x2)
        # with n_iters=3, there will be 3 figures. Put those figures in your written report. 
        
    
    return c

def classify(X,c):
    dist_1 = (X[0]-c[0][0])**2+(X[1]-c[1][0])**2
    dist_2 = (X[0]-c[0][1])**2+(X[1]-c[1][1])**2
    if dist_1>dist_2:
        return 0
    else:
        return 1

k_means(X=None, init_c=None, n_iters=3)