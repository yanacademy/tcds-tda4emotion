from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
import pickle
import numpy as np
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude, PairwiseDistance, Filtering
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import TakensEmbedding, SlidingWindow, takens_embedding_optimal_parameters
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def frequency_band(data, fs=128, fq=['delta', 'theta', 'alpha', 'beta', 'gamma']):
    wn = {'delta': [1 * 2 / fs, 4 * 2 / fs],
          'theta': [4 * 2 / fs, 8 * 2 / fs],
          'alpha': [8 * 2 / fs, 13 * 2 / fs],
          'beta': [13 * 2 / fs, 30 * 2 / fs],
          'gamma': [30 * 2 / fs, 50 * 2 / fs]
          }
    new_data = []
    for key, w in wn.items():
        if key in fq:
            [k, l] = butter(2, Wn=w, btype='bandpass')
            result = filtfilt(k, l, data)
            new_data.append(result)
    return np.array(new_data)


def PD(new_x, t, d):
    fq_pe = []
    for fq in new_x:  # 5fq
        steps = [TakensEmbedding(time_delay=t, dimension=d),
                 VietorisRipsPersistence(),
                 Scaler(),
                 Filtering(epsilon=0.1, homology_dimensions=(1, 2)),
                 PairwiseDistance(metric='wasserstein')
                 ]
        tda_pipe = make_pipeline(*steps)
        PE = tda_pipe.fit_transform(fq)
        #
        # feature = [('AM1', Amplitude(metric='bottleneck')),
        #            ('AM2', Amplitude(metric='wasserstein')),
        #            ('NOP', NumberOfPoints()),
        #            ('PE', PersistenceEntropy(normalize=True))]
        # feature_union = FeatureUnion(feature)
        # PE = feature_union.fit_transform(diagrams)
        fq_pe.append(PE)
    return fq_pe


def data_set(path, num_ch=0):
    with open(path, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
        x_train = []
        y_train = []
        for i in tqdm(range(0, 40)):
            data = subject['data'][i]
            labels = subject['labels'][i]
            optimal_time_delay, optimal_dimension = takens_embedding_optimal_parameters(
                data[num_ch], max_time_delay=12, max_dimension=5)
            sw = SlidingWindow(size=128, stride=64).fit_transform(data[num_ch].transpose())
            fqs = frequency_band(sw,fq=['gamma'])
            feature = np.concatenate(PD(fqs, optimal_time_delay, optimal_dimension), axis=1)
            x_train.append(feature)
            valence = 1 if labels[0] > 5 else 0
            arousal = 1 if labels[1] > 5 else 0
            y_train += [valence] * len(feature)
    return np.concatenate(x_train, axis=0), y_train


def training(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre, digits=4))


import os

files = [os.path.join('./dataset/', name) for name in os.listdir('./dataset/')]
X = []
Y = []
for file in files:
    print(file)
    tmp_x, tmp_y = data_set(file)
    training(tmp_x, tmp_y)

    X.append(tmp_x)
    Y += tmp_y
X = np.concatenate(X, axis=0)
training(X,Y)
# x_tr, y_tr = data_set('./dataset/s01.dat')
# x_te, y_te = data_set('./dataset/s02.dat')
# training(x_tr, x_te, y_tr, y_te)
#
# x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, test_size=0.3, shuffle=True)
# training(x_train, x_test, y_train, y_test)
#
# all_x = np.concatenate([x_tr, x_te], axis=0)
# all_y = np.concatenate([y_tr, y_te])
# x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.3, shuffle=True)
# training(x_train, x_test, y_train, y_test)
