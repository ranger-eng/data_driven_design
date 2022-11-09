from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


NP = 64;
NF=8; NF2=4; HL=np.zeros(NF);HH=np.zeros(NF);
N = NP+NF-1
N2 = N/2

print(NP,N,N2)

# Output
YL=np.zeros(N); YH=np.zeros(N);

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

# digits = load_digits()

# # pull training/testing data
# res = train_test_split(digits.data, digits.target, train_size=.8, test_size=.2, random_state=1)
# train_data, test_data, train_labels, test_labels = res

# mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)

# mlp.fit(train_data, train_labels)

# predictions = mlp.predict(test_data)
# a = accuracy_score(test_labels, predictions)
# print(a)

