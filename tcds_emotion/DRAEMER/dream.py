from collections import Counter
import plotly
from gtda.diagrams import Scaler, PairwiseDistance, PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import SlidingWindow, PearsonDissimilarity, TakensEmbedding
from scipy.io import loadmat
import numpy as np
import os
import warnings
import sklearn.svm as svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from tqdm import tqdm
from gtda.plotting import plot_point_cloud
from pandas import DataFrame
from pandas import Grouper
import matplotlib.pyplot as plt
#plt.style.use('seaborn')

warnings.filterwarnings("ignore")

# Draw Plot

def training_rf(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = RandomForestClassifier()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f using RF" % (scores.mean(), scores.std()))



def training_svm(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = svm.LinearSVC()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f using svm" % (scores.mean(), scores.std()))



def training_lr(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = LogisticRegression()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f using lr" % (scores.mean(), scores.std()))




def training_knn(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = KNeighborsClassifier()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f using knn" % (scores.mean(), scores.std()))


def training_gnb(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = GaussianNB()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f using gnb" % (scores.mean(), scores.std()))


def frequency_band(data, fs=128, fq=['theta', 'alpha', 'beta', 'gamma']):
    wn = {
        'theta': [4 * 2 / fs, 8 * 2 / fs],
        'alpha': [8 * 2 / fs, 13 * 2 / fs],
        'beta': [13 * 2 / fs, 30 * 2 / fs],
        'gamma': [30 * 2 / fs, 45 * 2 / fs]
    }
    new_data = []
    for key, w in wn.items():
        if key in fq:
            [k, l] = butter(2, Wn=w, btype='bandpass')
            result = filtfilt(k, l, data)
            new_data.append(result)
    return np.array(new_data)


def PD(new_x):
    fq_pe = []
    for fq in new_x:  # 4fq
        ##### plot code ###
        #sld = SlidingWindow(size=384, stride=384)
        #plottest = sld.fit_transform(fq)
        #tde = TakensEmbedding(time_delay=30, dimension=3)
        #pdforplot = tde.fit_transform(plottest)
        #fig = plot_point_cloud(pdforplot[0])
        #fig.show()
        #fig.write_image(os.getcwd() + "/learning.png") #调整下参数

        steps = [
            SlidingWindow(size=64, stride=64),
            TakensEmbedding(time_delay=5, dimension=3),
            VietorisRipsPersistence(),
            Scaler(),
            PersistenceLandscape(n_bins=50)
        ]
        tda_pipe = make_pipeline(*steps)
        PE = tda_pipe.fit_transform(fq)
        PE = np.mean(PE, axis=1)

        fq_pe.append(PE)
    return fq_pe


files = [os.path.join('./dataset', i) for i in os.listdir('./dataset')]
mov = ['Data' if i == 0 else 'Data' + str(i) for i in range(18)]
channel = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
all_x = []
all_a = []
all_v = []
all_d = []
for file in files:
    print(file)
    X = []
    ARS = []
    VLC = []
    DMN = []

    dreamer = loadmat(file)
    arousal = np.squeeze(dreamer['arousal'])
    valence = np.squeeze(dreamer['valence'])
    dominance = np.squeeze(dreamer['dominance'])
    for num, ch in tqdm(enumerate(mov)):
        data = dreamer[ch][0][0]

        tmp_feat = []
        #for i in range(6,7):
        #    df = data[1000:1500, i]
        #    plt.figure(figsize=(8, 2))
        #    plt.plot(df, color="green")
        #    plt.show()

        for i in range(14):
            eeg_data = data[:, i]
            fqs = frequency_band(eeg_data)
            #for fi in range(4):
            #    df = fqs[fi, 1372:1500]
            #    plt.figure(figsize=(4, 2))
            #    plt.plot(df, color="blue")
            #    plt.show()
            feature = np.concatenate(PD(fqs), axis=1)

            tmp_feat.append(feature)
        tmp_feat = np.concatenate(tmp_feat, axis=1)

        X.append(tmp_feat)
        v = 1 if valence[num] > 3 else 0
        a = 1 if arousal[num] > 3 else 0
        d = 1 if dominance[num] > 3 else 0

        VLC += [v] * len(tmp_feat)
        ARS += [a] * len(tmp_feat)
        DMN += [d] * len(tmp_feat)
    X = np.concatenate(X, axis=0)

    print(X.shape)
    print(dict(Counter(VLC)))
    print(dict(Counter(ARS)))
    print(dict(Counter(DMN)))
    print('all channel')
    training_rf(X, VLC)
    training_rf(X, ARS)
    training_rf(X, DMN)

    training_lr(X, VLC)
    training_lr(X, ARS)
    training_lr(X, DMN)

    training_svm(X, VLC)
    training_svm(X, ARS)
    training_svm(X, DMN)

    training_knn(X, VLC)
    training_knn(X, ARS)
    training_knn(X, DMN)

    training_gnb(X, VLC)
    training_gnb(X, ARS)
    training_gnb(X, DMN)


    all_x.append(X)
    all_v += VLC
    all_a += ARS
    all_d += DMN
# 每个channel和bands的结果
#   theta = []
#   alpha = []
#   beta = []
#   gamma = []
#   for i, ch_name in enumerate(channel):
#       ch_x = X[:, 200 * i:200 * (i + 1)]
#       print(f'feature shape:{ch_x.shape}.{ch_name}:')
#       training(ch_x, VLC)
#       training(ch_x, ARS)
#       training(ch_x, DMN)
#
#       theta.append(ch_x[:, :50])
#       alpha.append(ch_x[:, 50:100])
#       beta.append(ch_x[:, 100:150])
#       gamma.append(ch_x[:, 150:200])
#
#   print('theta:')
#   theta = np.concatenate(theta, axis=1)
#   training(theta, VLC)
#   training(theta, ARS)
#   training(theta, DMN)
#
#   print('alpha:')
#   alpha = np.concatenate(alpha, axis=1)
#   training(alpha, VLC)
#   training(alpha, ARS)
#   training(alpha, DMN)
#
#   print('beta:')
#   beta = np.concatenate(beta, axis=1)
#   training(beta, VLC)
#   training(beta, ARS)
#   training(beta, DMN)
#
#   print('gamma:')
#   gamma = np.concatenate(gamma, axis=1)
#   training(gamma, VLC)
#   training(gamma, ARS)
#   training(gamma, DMN)

print('final:')
all_x = np.concatenate(all_x, axis=0)
training_rf(all_x, all_v)
training_rf(all_x, all_a)
training_rf(all_x, all_d)

#np.save('./win2_x.npy', all_x)
#np.save('./win2_v.npy', all_v)
#np.save('./win2_a.npy', all_a)
#np.save('./win2_d.npy', all_d)