from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys


# Load the digits dataset
digits = datasets.load_digits()

N_total = digits.data.shape[0]
P = float(sys.argv[1])
N = int(np.round(N_total*P))
N_test = N_total - N
M = 64+1 # features
K = 10 #classes

phi = np.append(np.ones((1,N)), digits.data.T[:,0:N], axis=0)/100
t   = np.zeros((K,N))
for i in range(0,N,1):
    class_num = digits.target[i]
    t[class_num,i] = 1

wt  = np.random.rand(K,M)
a   = np.zeros((K,N))
y   = np.zeros((K,N), dtype=np.float128)
GRAD_E = np.zeros((M,K))

eta = .01*25

for ITER in range(0,20000,1):
    timer = time()
	#compute a's
    a= np.matmul(wt,phi)

	# compute soft-max
    for n in range(0,N,1):
        arg=0
        for k in range(0,K,1):
            arg = arg+ np.exp(a[k,n])
        for k in range(0,K,1):
            y[k,n]=np.exp(a[k,n])/arg

	#compute E
    E=0
    for k in range(0,K,1):
        for n in range(0,N,1):
            E=E - t[k,n]*np.log(y[k,n])

	#compute gradient BISHOP 4.109
    for k in range(0,K,1):
        for m in range(0,M,1):
            GRAD_E[m,k] = 0
            for n in range(0,N,1):
                GRAD_E[m,k] = GRAD_E[m,k]+(y[k,n]-t[k,n])*phi[m,n]

	# grad descent
    for k in range(0,K,1):
        for m in range(0,M,1):
            wt[k,m]  = wt[k,m] - eta*GRAD_E[m,k]

    # save weight
    if((ITER % 1000)==1):
        with open('weights{0}.npy'.format(ITER), 'wb') as f:
            np.save(f, wt)
            f.close()

    dt = time() - timer
    print(ITER,P,E,dt)


