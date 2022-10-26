from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from time import time

M = 64+1 # features
K = 10 #classes

# Load the digits dataset
digits = datasets.load_digits()
# determine number of samples in each class
N0 = 0; N1 = 0; N2 = 0; N3 = 0; N4 = 0; N5 = 0; N6 = 0; N7 = 0; N8 = 0; N9 = 0
for i in range(0,np.shape(digits.target)[0],1):
    if digits.target[i] == 0:
        N0 = N0+1
    if digits.target[i] == 1:
        N1 = N1+1
    if digits.target[i] == 2:
        N2 = N2+1
    if digits.target[i] == 3:
        N3 = N3+1
    if digits.target[i] == 4:
        N4 = N4+1
    if digits.target[i] == 5:
        N5 = N5+1
    if digits.target[i] == 6:
        N6 = N6+1
    if digits.target[i] == 7:
        N7 = N7+1
    if digits.target[i] == 8:
        N8 = N8+1
    if digits.target[i] == 9:
        N9 = N9+1
N = N0+N1+N2+N3+N4+N5+N6+N7+N8+N9

RAW0 = np.zeros((M,N0)); t0 = np.zeros((K,N0))
RAW1 = np.zeros((M,N1)); t1 = np.zeros((K,N1))
RAW2 = np.zeros((M,N2)); t2 = np.zeros((K,N2))
RAW3 = np.zeros((M,N3)); t3 = np.zeros((K,N3))
RAW4 = np.zeros((M,N4)); t4 = np.zeros((K,N4))
RAW5 = np.zeros((M,N5)); t5 = np.zeros((K,N5))
RAW6 = np.zeros((M,N6)); t6 = np.zeros((K,N6))
RAW7 = np.zeros((M,N7)); t7 = np.zeros((K,N7))
RAW8 = np.zeros((M,N8)); t8 = np.zeros((K,N8))
RAW9 = np.zeros((M,N9)); t9 = np.zeros((K,N9))
i0 = 0; i1 = 0; i2 = 0; i3 = 0; i4 = 0; i5 = 0; i6 = 0; i7 = 0; i8 = 0; i9 = 0
for i in range(0,np.shape(digits.target)[0],1):
    if digits.target[i] == 0:
        RAW0[0,i0] = 1
        for n in range(0,M-1,1):
            RAW0[n+1,i0] = digits.data[i,n]
        t0[0,i0] = 1
        i0 = i0+1
    if digits.target[i] == 1:
        RAW1[0,i1] = 1
        for n in range(0,M-1,1):
            RAW1[n+1,i1] = digits.data[i,n]
        t1[1,i1] = 1
        i1 = i1+1
    if digits.target[i] == 2:
        RAW2[0,i2] = 1
        for n in range(0,M-1,1):
            RAW2[n+1,i2] = digits.data[i,n]
        t2[2,i2] = 1
        i2 = i2+1
    if digits.target[i] == 3:
        RAW3[0,i3] = 1
        for n in range(0,M-1,1):
            RAW3[n+1,i3] = digits.data[i,n]
        t3[3,i3] = 1
        i3 = i3+1
    if digits.target[i] == 4:
        RAW4[0,i4] = 1
        for n in range(0,M-1,1):
            RAW4[n+1,i4] = digits.data[i,n]
        t4[4,i4] = 1
        i4 = i4+1
    if digits.target[i] == 5:
        RAW5[0,i5] = 1
        for n in range(0,M-1,1):
            RAW5[n+1,i5] = digits.data[i,n]
        t5[5,i5] = 1
        i5 = i5+1
    if digits.target[i] == 6:
        RAW6[0,i6] = 1
        for n in range(0,M-1,1):
            RAW6[n+1,i6] = digits.data[i,n]
        t6[6,i6] = 1
        i6 = i6+1
    if digits.target[i] == 7:
        RAW7[0,i7] = 1
        for n in range(0,M-1,1):
            RAW7[n+1,i7] = digits.data[i,n]
        t7[7,i7] = 1
        i7 = i7+1
    if digits.target[i] == 8:
        RAW8[0,i8] = 1
        for n in range(0,M-1,1):
            RAW8[n+1,i8] = digits.data[i,n]
        t8[8,i8] = 1
        i8 = i8+1
    if digits.target[i] == 9:
        RAW9[0,i9] = 1
        for n in range(0,M-1,1):
            RAW9[n+1,i9] = digits.data[i,n]
        t9[9,i9] = 1
        i9 = i9+1


wt  = np.random.rand(K,M)
phi = np.empty((M,0))
a   = np.zeros((K,N))
t   = np.empty((K,0))
y   = np.zeros((K,N), dtype=np.float128)
GRAD_E = np.zeros((M,K))

eta = .01*6

# populate phi and t
phi = np.append(phi,RAW0,axis=1); t = np.append(t,t0,axis=1)
phi = np.append(phi,RAW1,axis=1); t = np.append(t,t1,axis=1)
phi = np.append(phi,RAW2,axis=1); t = np.append(t,t2,axis=1)
phi = np.append(phi,RAW3,axis=1); t = np.append(t,t3,axis=1)
phi = np.append(phi,RAW4,axis=1); t = np.append(t,t4,axis=1)
phi = np.append(phi,RAW5,axis=1); t = np.append(t,t5,axis=1)
phi = np.append(phi,RAW6,axis=1); t = np.append(t,t6,axis=1)
phi = np.append(phi,RAW7,axis=1); t = np.append(t,t7,axis=1)
phi = np.append(phi,RAW8,axis=1); t = np.append(t,t8,axis=1)
phi = np.append(phi,RAW9,axis=1); t = np.append(t,t9,axis=1)
phi = phi/100

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
    print(ITER,E,dt,np.shape(wt)[0],np.shape(wt)[1],np.unwrap(wt))


