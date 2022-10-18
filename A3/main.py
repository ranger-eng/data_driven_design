import numpy as np
import matplotlib.pyplot as plt
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
N0 = 25
N1 = 25
N2 = 25

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

#################
# HW 4
# c. consider the LDA approach using the same data. Due to strong 
# correlation between features 2 and 0 in class 1 plot the scatter 
# of all classes in th 0-2 feature plane.

# features 0 and 2 for class 0
temp0 = pd.read_csv("./data/hw3_class0.dat", sep=" ", header=None).to_numpy()[:,[0,2]]
temp1 = pd.read_csv("./data/hw3_class1.dat", sep=" ", header=None).to_numpy()[:,[0,2]]
temp2 = pd.read_csv("./data/hw3_class2.dat", sep=" ", header=None).to_numpy()[:,[0,2]]

f0_c0 = temp0[:,0]
f2_c0 = temp0[:,1]
f0_c1 = temp1[:,0]
f2_c1 = temp1[:,1]
f0_c2 = temp2[:,0]
f2_c2 = temp2[:,1]
plt.scatter(f0_c0,f2_c0,c="red")
plt.scatter(f0_c1,f2_c1,c="blue")
plt.scatter(f0_c2,f2_c2,c="green")
plt.legend(["class0","class1","class2"])
plt.title("Scatter plot of three classes in feature 0, feature 2 plane")
# plt.show()

#################
# compute SW
mean_0 = np.mean(X0,axis=1)
mean_1 = np.mean(X1,axis=1)
mean_2 = np.mean(X2,axis=1)
mean_total = 1/O*(N0*mean_0 + N1*mean_1 + N2*mean_2)
print(mean_0, mean_1, mean_2, mean_total)

X0_minus_mean = np.zeros([M,N0])
X1_minus_mean = np.zeros([M,N1])
X2_minus_mean = np.zeros([M,N2])

for i in range(0,M,1):
    for j in range(0,N0,1):
        X0_minus_mean[i,j] = X0[i,j]-mean_0[i]
    for j in range(0,N1,1):
        X1_minus_mean[i,j] = X1[i,j]-mean_1[i]
    for j in range(0,N2,1):
        X2_minus_mean[i,j] = X2[i,j]-mean_2[i]

S0 = np.dot(X0_minus_mean,X0_minus_mean.T)
S1 = np.dot(X1_minus_mean,X1_minus_mean.T)
S2 = np.dot(X2_minus_mean,X2_minus_mean.T)

SW = S0+S1+S2

#################
# compute SB
SB = np.zeros([M,M])
for i in range(0,M,1):
    for j in range(0,M,1):
        SB[i,j] = N0*(mean_0[i]-mean_total[i])*(mean_0[j]-mean_total[j]) + \
        N1*(mean_1[i]-mean_total[i])*(mean_1[j]-mean_total[j]) + \
        N2*(mean_2[i]-mean_total[i])*(mean_2[j]-mean_total[j])

#################
# find the eigenvalues and eigenvectors of SW^-1*SB
SW_inv = np.linalg.solve(SW,np.identity(M))
SW_inv_SB = np.dot(SW_inv,SB)

J,W_LDA = np.linalg.eig(SW_inv_SB)
J = J[0]
W_LDA = W_LDA[:,0]

print("J:",J,"W_LDA:",W_LDA)

R_LDA = np.dot(W_LDA,X)
confused_LDA = np.zeros([N,N])
for i in range (0,O,1):
	#predicted
	arg= np.amax (R_LDA[i])
	tmp= np.where(R_LDA[i]==np.amax(R_LDA[i]))
	qr= tmp[0]
	#actual
	arg= np.amax (T[:,i])
	tmp= np.where(T[:,i]==np.amax(T[:,i]))
	qt= tmp[0]

	confused_LDA[qr[0],qt[0]]= confused_LDA[qr[0],qt[0]]+1
	#print(qt[0],qr[0])

print('actual horizontal vs predicted vertical');
for i in range (0,N,1):
	print( confused_LDA[i,:]/25.0)
