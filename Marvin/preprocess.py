from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

###########
# load data
marvin = wavfile.read("./wav/Marvin.wav")
bass = wavfile.read("./wav/Bass.wav")
drum1 = wavfile.read("./wav/Drumstam.wav")
drum2 = wavfile.read("./wav/Drums.wav")
piano = wavfile.read("./wav/Piano_Bell.wav")
strings = wavfile.read("./wav/Strings.wav")
tammi = wavfile.read("./wav/Tammi.wav")
al = wavfile.read("./wav/All.wav")
# al[1] = marvin[1] + bass[1]/2 + drum1[1]/2 + drum2[1]/2 + piano[1]/2 + strings[1]/2 + tammi[1]/2

# problem setup
Fs = 44100                             # sample rate
K = 2                                  # classes
Seg_len = 1024*2                         # number of time domain samples per chunk
N_train = int(marvin[1].shape[0]/Seg_len) \
+ int(bass[1].shape[0]/Seg_len) \
+ int(drum1[1].shape[0]/Seg_len) \
+ int(drum2[1].shape[0]/Seg_len) \
+ int(piano[1].shape[0]/Seg_len) \
+ int(strings[1].shape[0]/Seg_len) \
+ int(tammi[1].shape[0]/Seg_len) \

N_test = int(al[1].shape[0]/Seg_len)      # testing samples

# test out creating a spectrogram
nfft_no = 512
win_no = 0
x = al[1][win_no*Seg_len:(win_no+1)*Seg_len]
f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
Sxx = Sxx.flatten()
M = Sxx.shape[0]                       # number of features

# define matrix shapes
train_data  = np.zeros([N_train, M])
train_label = np.zeros([N_train,])
test_data   = np.zeros([N_test, M])
test_label  = np.zeros([N_test,])

###########
# populate matricies
# loop over marvin data in Seg_len intervals
threshold = 100000
for i in range(0, N_test):
    x_marvin = marvin[1][i*Seg_len:(i+1)*Seg_len]
    x_al = al[1][i*Seg_len:(i+1)*Seg_len]

    magnitude = np.linalg.norm(x_marvin)
    print(magnitude)

    # if the magnitude is larger than the threshold then there is speech
    # label test_label and train_label
    if (magnitude > threshold):
        train_label[i] = 1
        test_label[i]  = 1
    else:
        train_label[i] = 0
        test_label[i] = 0

    # populate data matricies with the spectrogram
    f, t, Sxx_marvin = spectrogram(x_marvin, Fs, nfft=nfft_no)
    Sxx_marvin = Sxx_marvin.flatten()
    train_data[i] = Sxx_marvin

    f, t, Sxx_al = spectrogram(x_al, Fs, nfft=nfft_no)
    Sxx_al = Sxx_al.flatten()
    test_data[i] = Sxx_al

# bass
wav_file_no = 1
for i in range(0, N_test):
    x = bass[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

# drum1 = wavfile.read("./wav/Drumstam.wav")
wav_file_no = 2
for i in range(0, N_test):
    x = drum1[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

# drum2 = wavfile.read("./wav/Drums.wav")
wav_file_no = 3
for i in range(0, N_test):
    x = drum2[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

# piano = wavfile.read("./wav/Piano_Bell.wav")
wav_file_no = 4
for i in range(0, N_test):
    x = piano[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

# strings = wavfile.read("./wav/Strings.wav")
wav_file_no = 5
for i in range(0, N_test):
    x = strings[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

# tammi = wavfile.read("./wav/Tammi.wav")
wav_file_no = 6
for i in range(0, N_test):
    x = tammi[1][i*Seg_len:(i+1)*Seg_len]

    train_label[i+N_test*wav_file_no] = 0

    f, t, Sxx = spectrogram(x, Fs, nfft=nfft_no)
    Sxx = Sxx.flatten()
    train_data[i+N_test*wav_file_no] = Sxx

#############
# train and test

#############
# save and load data
np.save('./test_data', test_data)
np.save('./train_data', train_data)
np.save('./test_label', test_label)
np.save('./train_label', train_label)

test_data = np.load('./test_data.npy')
train_data = np.load('./train_data.npy')
test_label = np.load('./test_label.npy')
train_label = np.load('./train_label.npy')

hidden_layers = 100
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=20000, activation='logistic', alpha=1e-5, solver='lbfgs', tol=1e-6, random_state=1, learning_rate_init=.02, verbose=True)

mlp.fit(train_data, train_label)

with open('mlp_lbfgs_100hidden.pickle', 'wb') as handle:
    pickle.dump(mlp, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("testing DNN")
predictions = mlp.predict(test_data)

acc = accuracy_score(test_label, predictions)
a = confusion_matrix(predictions,test_label, labels=[0,1], normalize='true')

ax = sns.heatmap(a, linewidth=0.5, vmin=0, vmax=1)
plt.title("Marvin_DNN {0:.02f}% accuracy".format(acc*100))
plt.savefig('./figures/mlp_lbgfs_100hidden.png', dpi=100)
plt.clf()
