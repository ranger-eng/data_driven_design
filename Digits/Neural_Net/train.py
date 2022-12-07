from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time

digits = load_digits()

# pull training/testing data
P = np.array([.10, .25, .50, .75])


for i in range(0,P.size,1):
    timer = time()
    P_train = P[i]
    P_test = 1 - P[i]
    res = train_test_split(digits.data, digits.target, train_size=P_train, test_size=P_test, random_state=1)
    train_data, test_data, train_labels, test_labels = res

    hidden_layers = 400
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)

    # mlp.fit(train_data, train_labels)

    # predictions = mlp.predict(test_data)
    # dt = time() - timer

    # acc = accuracy_score(test_labels, predictions)
    # a = confusion_matrix(predictions,test_labels, labels=[0,1,2,3,4,5,6,7,8,9], normalize='true')

    # ax = sns.heatmap(a, linewidth=0.5, vmin=0, vmax=1)
    # plt.title("NeuralNet - {2} Layers - {0}% Training - Accuracy={1:.2f}% in {3:.1f}s".format(100*P_train,100*acc,hidden_layers,dt))
    # plt.savefig('./results/nn_p{0}training_{1}hiddenlayers.png'.format(round(P[i]*100),hidden_layers), dpi=100)
    # plt.clf()


