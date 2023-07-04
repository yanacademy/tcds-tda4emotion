import math
import os
import warnings
import pandas as pd
from gtda.diagrams import PersistenceEntropy, Scaler, PairwiseDistance, PersistenceLandscape, Amplitude, Silhouette
from gtda.pipeline import make_pipeline
from gtda.time_series import SlidingWindow, TakensEmbedding, PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
warnings.filterwarnings("ignore")


def training(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre, digits=4))


steps = [
    # TakensEmbedding(time_delay=10, dimension=5),
    VietorisRipsPersistence(),
    Scaler(),
    PairwiseDistance(metric='wasserstein')
]
file = 'dataset/s01.dat'

with open(file, 'rb') as f:
    subject = pickle.load(f, encoding='latin1')
    y = []
    PEAR = []
    for i in tqdm(range(0, 40)):
        eeg_data = subject['data'][i]
        label = subject['labels'][i]
        eeg_data = eeg_data[:32]
        sw = SlidingWindow(size=256, stride=128).fit_transform(eeg_data.transpose())
        pearson = PearsonDissimilarity().fit_transform(sw)
        PEAR.append(pearson)

        valence = 1 if label[0] > 5 else 0
        arousal = 1 if label[1] > 5 else 0
        y += [arousal] * len(sw)

    PEAR = np.concatenate(PEAR, axis=0)

    state = np.random.get_state()
    np.random.shuffle(PEAR)
    np.random.set_state(state)
    np.random.shuffle(y)
    tda_pipe = make_pipeline(*steps)
    print('get distance matrix')
    matrix = tda_pipe.fit_transform(PEAR)

    np.savetxt('./v_list.csv', y, delimiter=',')

    totaln = matrix.shape[0]
    for j in range(0, totaln):
        for i in range(0, j):
            matrix[i][j] = matrix[j][i]

    k = 60
    trainnum = math.floor(0.8 * totaln)
    print(trainnum)
    trainlist = [i for i in range(0, trainnum)]

    testnum = totaln - trainnum
    testlist = [i for i in range(trainnum, totaln)]

    knn = [[] for i in range(testnum)]

    for i in range(0, testnum):
        index = testlist[i]

        coltobesort = matrix[0:trainnum, index]
        indexsorted = np.argsort(coltobesort)
        indexsorted = indexsorted[indexsorted != index]

        knn[i] = indexsorted[0:k]

    df = pd.read_csv('./v_list.csv', delimiter=',', header=None)
    df.columns = ['class']
    y_ture = df.iloc[testlist]
    y_ture = y_ture.values.tolist()
    y_pred = []
    for i in range(0, len(knn)):
        knnclass = df['class'].iloc[knn[i]]
        totalsum = knnclass.sum()
        average = totalsum / k
        if (average < 0.5):
            y_pred.append(0)
        else:
            y_pred.append(1)

    print('predict results:')
    print('y_pred:')
    print(y_pred)
    target_names = ['low', 'high']
    print(classification_report(y_ture, y_pred, labels=[0, 1],
                                target_names=target_names, digits=4))

