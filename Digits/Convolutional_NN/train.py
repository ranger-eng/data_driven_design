from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


NP = 64;
NF=8; NF2=4; HL=np.zeros(NF);HH=np.zeros(NF);
N = NP+NF-1
N2 = int(round(N/2))

print(NP,N,N2)

# Decimated (by 2) Output
YL2=np.zeros(N2); YH2=np.zeros(N2);

##############################
# FIR FILTER coef
# Low -pass:
HL[4]= 0.48998080
HL[5]= 0.69428270/10.0
HL[6]=-0.70651830/10.0
HL[7]= 0.93871500/100.0
for i in range (0,NF2,1):
	HL[i] = HL[NF-1-i]
# High-pass:
arg = 1.
for i in range (0,NF,1):
    HH[i] = HL[i]*arg
    arg = -arg
    print(i,HL[i],HH[i])

###############

digits = load_digits()

# pull training/testing data
res = train_test_split(digits.data, digits.target, train_size=.8, test_size=.2, random_state=1)
train_data, test_data, train_labels, test_labels = res

# create empty arrays
train_data_HL = np.zeros([np.shape(train_data)[0],N])
train_data_HH = np.zeros([np.shape(train_data)[0],N])
train_data_HL2 = np.zeros([np.shape(train_data)[0],N2])
train_data_HH2 = np.zeros([np.shape(train_data)[0],N2])
train_data_HLHH = np.zeros([np.shape(train_data)[0],N2*2])

test_data_HL = np.zeros([np.shape(test_data)[0],N])
test_data_HH = np.zeros([np.shape(test_data)[0],N])
test_data_HL2 = np.zeros([np.shape(test_data)[0],N2])
test_data_HH2 = np.zeros([np.shape(test_data)[0],N2])
test_data_HLHH = np.zeros([np.shape(test_data)[0],N2*2])

# filter data
for i in range(0,np.shape(train_data)[0]):
    train_data_HL[i] = np.convolve(train_data[i],HL)
    train_data_HH[i] = np.convolve(train_data[i],HH)

for i in range(0,np.shape(test_data)[0]):
    test_data_HL[i] = np.convolve(test_data[i],HL)
    test_data_HH[i] = np.convolve(test_data[i],HH)

# ditch every other sample
for i in range(0,np.shape(train_data)[0]):
    for j in range(0,np.shape(train_data_HL)[1]):
        if ((j+1) % 2):
            train_data_HL2[i,int(round(j/2))] = train_data_HL[i,j]
            train_data_HH2[i,int(round(j/2))] = train_data_HH[i,j]

for i in range(0,np.shape(test_data)[0]):
    for j in range(0,np.shape(test_data_HL)[1]):
        if ((j+1) % 2):
            test_data_HL2[i,int(round(j/2))] = test_data_HL[i,j]
            test_data_HH2[i,int(round(j/2))] = test_data_HH[i,j]

# combine HL and HH
for i in range(0,np.shape(train_data)[0]):
    for j in range(0,np.shape(train_data_HL2)[1]):
        train_data_HLHH[i,j] = train_data_HL2[i,j]
        train_data_HLHH[i,j+N2-1] = train_data_HH2[i,j]

for i in range(0,np.shape(test_data)[0]):
    for j in range(0,np.shape(test_data_HL2)[1]):
        test_data_HLHH[i,j] = test_data_HL2[i,j]
        test_data_HLHH[i,j+N2-1] = test_data_HH2[i,j]


# Full Dataset
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)
mlp.fit(train_data, train_labels)
predictions = mlp.predict(test_data)
a = accuracy_score(test_labels, predictions)

# HH
mlp_HH = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)
mlp_HH.fit(train_data_HH2, train_labels)
predictions_HH = mlp_HH.predict(test_data_HH2)
a_HH = accuracy_score(test_labels, predictions_HH)

# HL
mlp_HL = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)
mlp_HL.fit(train_data_HL2, train_labels)
predictions_HL = mlp_HL.predict(test_data_HL2)
a_HL = accuracy_score(test_labels, predictions_HL)

# HLHH
mlp_HLHH = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)
mlp_HLHH.fit(train_data_HLHH, train_labels)
predictions_HLHH = mlp_HLHH.predict(test_data_HLHH)
a_HLHH = accuracy_score(test_labels, predictions_HLHH)

print("High Pass Accuracy: ",a_HH,", Low Pass Accuracy: ",a_HL, ", Combined Highpass Lowpass Accuracy:",a_HLHH, ", Full Dataset no Convolution: ",a)
