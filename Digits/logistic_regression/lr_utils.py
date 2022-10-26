import numpy as np
from sklearn import datasets

def logistic_mult(wt, phi):
    K = wt.shape[0]
    M = wt.shape[1]
    N = phi.shape[1]

    if (M != phi.shape[0]):
        raise ValueError("wt and phi matrix mismatched inner dimension.")

    y = np.zeros((K,N), dtype=np.float128)

    # compute a's
    a = np.matmul(wt,phi)

    # compute soft-max
    for n in range(0,N,1):
        arg=0
        for k in range(0,K,1):
            arg = arg+ np.exp(a[k,n])
        for k in range(0,K,1):
            y[k,n]=np.exp(a[k,n])/arg

    return y

def populate_phi(P):
    ''' P is a decimal representing the amount of training data. ex .10 for 10% training, 90% testing '''
    ''' returns phi_train, phi_test '''
    digits = datasets.load_digits()

    N_total = digits.data.shape[0]
    N = int(np.round(N_total*P))
    N_test = N_total - N

    M = 64+1
    K = 10

    phi_train = np.append(np.ones((1,N)), digits.data.T[:,0:N], axis=0)/100
    phi_test = np.append(np.ones((1,N_test)), digits.data.T[:,N:N_total], axis=0)/100

    return phi_train, phi_test

def interpret_y(y, P):
    ''' the max value in each column is the prediction of the class '''
    ''' there are 10 classes for the digits problem, 0-9 '''
    digits = datasets.load_digits()

    N_total = digits.data.shape[0]
    N = int(np.round(N_total*P))
    N_test = N_total - N

    M = 64+1
    K = 10

    result = np.zeros(y.shape[1])

    for i in range(0,y.shape[1],1):
        max_val = np.max(y[:,i])
        for j in range(0,y.shape[0],1):
            if (y[j,i] == max_val):
                result[i] = j
    
    ground_truth = digits.target[N:N_total]

    return result, ground_truth
