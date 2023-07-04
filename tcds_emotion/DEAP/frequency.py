from collections import Counter

from sklearn.pipeline import FeatureUnion
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle
import numpy as np
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude, NumberOfPoints, PairwiseDistance, PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import TakensEmbedding, SlidingWindow
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import os


def frequency_band(data, fs=128, fq=['theta', 'alpha', 'beta', 'gamma']):
    wn = {'delta': [1 * 2 / fs, 4 * 2 / fs],
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


def raw_PD(sw):
    steps = [
        SlidingWindow(size=128,stride=128),
        TakensEmbedding(time_delay=10, dimension=8),
        VietorisRipsPersistence(),
        Scaler(),
        PersistenceLandscape(n_bins=50)
    ]
    tda_pipe = make_pipeline(*steps)
    PE = tda_pipe.fit_transform(sw)
    PE = np.mean(PE, axis=1)
    return PE


def PD(new_x):
    fq_pe = []
    for fq in new_x:  # 5fq
        steps = [
            SlidingWindow(size=128, stride=128),
            TakensEmbedding(time_delay=10, dimension=8),
            VietorisRipsPersistence(),
            Scaler(),
            PersistenceLandscape(n_bins=50)
        ]
        tda_pipe = make_pipeline(*steps)
        PE = tda_pipe.fit_transform(fq)
        PE = np.mean(PE, axis=1)
        #
        # feature = [('AM1', Amplitude(metric='bottleneck')),
        #            ('AM2', Amplitude(metric='wasserstein'))]
        # feature_union = FeatureUnion(feature)
        # PE = feature_union.fit_transform(PE)
        fq_pe.append(PE)
    return fq_pe


def data_set(path, num_ch=0):
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
        x_train = []
        A = []
        V = []
        for i in tqdm(range(0, 40)):
            data = subject['data'][i]
            labels = subject['labels'][i]

            ch_feat = []
            for ch in range(32):
                fqs = frequency_band(data[ch])
                feature = np.concatenate(PD(fqs), axis=1)
                ch_feat.append(feature)
            ch_feat = np.concatenate(ch_feat, axis=1)

            x_train.append(ch_feat)

            valence = 1 if labels[0] > 5 else 0
            arousal = 1 if labels[1] > 5 else 0

            V += [valence] * len(ch_feat)
            A += [arousal] * len(ch_feat)
    return np.concatenate(x_train, axis=0), V, A


def training(x, y):
    shuffled_X, shuffled_Y = shuffle(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = RandomForestClassifier()
    # model.fit(x_train, y_train)
    # y_pre = model.predict(x_test)
    # print(classification_report(y_test, y_pre, digits=4))
    scores = cross_val_score(model, shuffled_X, shuffled_Y, cv=10)
    print(scores)
    print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))


channel = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
           'O1', 'Oz', 'Pz', 'FP2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
           'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
files = [os.path.join('./dataset/', name) for name in os.listdir('./dataset/')]
X = []
YV = []
YA = []
for file in files:
    print(file)
    tmp_x, tmp_v, tmp_a = data_set(file)

    print(tmp_x.shape)
    print(dict(Counter(tmp_v)))
    print(dict(Counter(tmp_a)))
    #
    # theta = []
    # alpha = []
    # beta = []
    # gamma = []
    # for i, ch_name in enumerate(channel):
    #     ch_x = tmp_x[:, 200 * i:200 * (i + 1)]
    #     print(f'feature shape:{ch_x.shape}.{ch_name}:')
    #     training(ch_x, tmp_v)
    #     training(ch_x, tmp_a)
    #
    #     theta.append(ch_x[:, :50])
    #     alpha.append(ch_x[:, 50:100])
    #     beta.append(ch_x[:, 100:150])
    #     gamma.append(ch_x[:, 150:200])
    #
    # print('theta:')
    # theta = np.concatenate(theta, axis=1)
    # training(theta, tmp_v)
    # training(theta, tmp_a)
    #
    # print('alpha:')
    # alpha = np.concatenate(alpha, axis=1)
    # training(alpha, tmp_v)
    # training(alpha, tmp_a)
    #
    # print('beta:')
    # beta = np.concatenate(beta, axis=1)
    # training(beta, tmp_v)
    # training(beta, tmp_a)
    #
    # print('gamma:')
    # gamma = np.concatenate(gamma, axis=1)
    # training(gamma, tmp_v)
    # training(gamma, tmp_a)
    #
    print('all channel:')
    training(tmp_x, tmp_v)
    training(tmp_x, tmp_a)
    X.append(tmp_x)
    YV += tmp_v
    YA += tmp_a
X = np.concatenate(X, axis=0)

training(X, YV)
training(X, YA)
