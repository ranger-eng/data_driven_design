import numpy as np
import matplotlib as plt
import pandas as pd

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# problem setup:
#
# N=3 classes
# M=4 features
# O=25*3=75 training vectors
#
# Training:
# setup the following matrix equation, W is unknown
#
# T[NxO] = W[NxM] * X[MxO]
N = 3
M = 4
O = 75

T = np.zeros([N,O])
X = np.array([])

# pull data from dataset
temp1 = pd.read_csv("./data/hw3_class0.dat", sep=" ", header=None).to_numpy()[0:25,:]
temp2 = pd.read_csv("./data/hw3_class1.dat", sep=" ", header=None).to_numpy()[0:25,:]
temp3 = pd.read_csv("./data/hw3_class2.dat", sep=" ", header=None).to_numpy()[0:25,:]
temp4 = np.append(temp1,temp2,axis=0)
X = np.append(temp4,temp3,axis=0).T

# set training flags for the correct class
for i in range(0,25):
    T[0,i] = 1
    T[1,i+25] = 1
    T[2,i+50] = 1

# compute W.T that minimizes the trace
term1 = np.dot(X,X.T)
term1 = np.linalg.inv(term1)
term2 = np.dot(X,T.T)
W = np.dot(term1,term2)
W = W.T

R = np.dot(W,X)
confused = np.zeros([N,N])
for i in range (0,O,1):
	#predicted
	arg= np.amax (R[:,i])
	tmp= np.where(R[:,i]==np.amax(R[:,i]))
	qr= tmp[0]
	#actual
	arg= np.amax (T[:,i])
	tmp= np.where(T[:,i]==np.amax(T[:,i]))
	qt= tmp[0]

	confused[qr[0],qt[0]]= confused[qr[0],qt[0]]+1
	#print(qt[0],qr[0])

print('actual horizontal vs predicted vertical');
for i in range (0,N,1):
	print( confused[i,:]/25.0)

#################################
# Test model using the rest of the data
temp0 = pd.read_csv("./data/hw3_class0.dat", sep=" ", header=None).to_numpy()[24:50,:]
temp1 = pd.read_csv("./data/hw3_class1.dat", sep=" ", header=None).to_numpy()[24:50,:]
temp2 = pd.read_csv("./data/hw3_class2.dat", sep=" ", header=None).to_numpy()[24:50,:]
X0 = temp0.T
X1 = temp1.T
X2 = temp2.T

Y0 = np.dot(W,X0)
Y1 = np.dot(W,X1)
Y2 = np.dot(W,X2)

correct0 = 0
for i in range (0,25,1):
    arg = np.amax(Y0[:,i])
    tmp = np.where(Y0[:,i]==arg)
    if (tmp[0] == 0):
        correct0 = correct0 + 1
print("Using least squares method for training, class0 has accuracy: ", correct0/25)

correct1 = 0
for i in range (0,25,1):
    arg = np.amax(Y1[:,i])
    tmp = np.where(Y1[:,i]==arg)
    if (tmp[0] == 1):
        correct1 = correct1 + 1
print("Using least squares method for training, class1 has accuracy: ", correct1/25)

correct2 = 0
for i in range (0,25,1):
    arg = np.amax(Y2[:,i])
    tmp = np.where(Y2[:,i]==arg)
    if (tmp[0] == 2):
        correct2 = correct2 + 1
print("Using least squares method for training, class2 has accuracy: ", correct2/25)
