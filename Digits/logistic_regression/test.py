import lr_utils
from sklearn import datasets
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv

directory = './results/'
lines = 20000

P = np.array([.10,.25,.50,.75])
for i in range(0,P.size,1):
    # load weights
    wt = np.load('./training/p{0}percent_training_data/weights19001.npy'.format(round(P[i]*100)))

    # populate phi with the remaining sklearn data
    phi_train, phi_test = lr_utils.populate_phi(P[i])

    # test data
    y = lr_utils.logistic_mult(wt, phi_test)
    result, ground_truth = lr_utils.interpret_y(y, P[i])

    a = metrics.confusion_matrix(result,ground_truth, labels=[0,1,2,3,4,5,6,7,8,9], normalize='true')
    acc = metrics.accuracy_score(result, ground_truth)

    ax = sns.heatmap(a, linewidth=0.5, vmin=0, vmax=1)
    plt.title("Logistic Regression - {0}% Training Data - Accuracy = {1:.2f}%".format(100*P[i],100*acc))
    plt.savefig('{0}logistic_regression_p{1}training.png'.format(directory,round(P[i]*100)), dpi=100)
    plt.clf()

##################################################################################################
P = .10
seq_p10 = np.zeros(lines)
E_p10 = np.zeros(lines)
dt_p10 = np.zeros(lines)
i = 0
with open('./training/p{0}percent_training_data/out.txt'.format(round(P*100)), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        seq_p10[i] = row[0]
        E_p10[i] = row[2]
        dt_p10[i] = row[3]
        i = i+1

P = .25
seq_p25 = np.zeros(lines)
E_p25 = np.zeros(lines)
dt_p25 = np.zeros(lines)
i = 0
with open('./training/p{0}percent_training_data/out.txt'.format(round(P*100)), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        seq_p25[i] = row[0]
        E_p25[i] = row[2]
        dt_p25[i] = row[3]
        i = i+1

P = .50
seq_p50 = np.zeros(lines)
E_p50 = np.zeros(lines)
dt_p50 = np.zeros(lines)
i = 0
with open('./training/p{0}percent_training_data/out.txt'.format(round(P*100)), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        seq_p50[i] = row[0]
        E_p50[i] = row[2]
        dt_p50[i] = row[3]
        i = i+1

P = .75
seq_p75 = np.zeros(lines)
E_p75 = np.zeros(lines)
dt_p75 = np.zeros(lines)
i = 0
with open('./training/p{0}percent_training_data/out.txt'.format(round(P*100)), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        seq_p75[i] = row[0]
        E_p75[i] = row[2]
        dt_p75[i] = row[3]
        i = i+1

plt.plot(seq_p10*np.mean(dt_p10)/60/60,E_p10/E_p10[0]*100, seq_p25*np.mean(dt_p25)/60/60,E_p25/E_p25[0]*100, seq_p50*np.mean(dt_p50)/60/60,E_p50/E_p50[0]*100, seq_p75*np.mean(dt_p75)/60/60, E_p75/E_p75[0]*100)
plt.xscale('log')
plt.yscale('log')
plt.title("Logistric Regression Convergence Rate")
plt.legend(["10% training data","25% training data","50% training data","75% training data"])
plt.ylabel("Error")
plt.xlabel("Time [hrs]")
plt.savefig('{0}logistic_regression_convergence_time.png'.format(directory,round(P*100)), dpi=100)
plt.clf()

