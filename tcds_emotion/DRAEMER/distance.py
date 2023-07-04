import math
import eegraph
import pandas as pd
from gtda.diagrams import PairwiseDistance, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import SlidingWindow, TakensEmbedding, PearsonDissimilarity
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os

import warnings

warnings.filterwarnings("ignore")


def PD(new_x):
    steps = [
        VietorisRipsPersistence(),
        Scaler(),
        PairwiseDistance(metric='wasserstein')
    ]
    tda_pipe = make_pipeline(*steps)
    distmatrix = tda_pipe.fit_transform(new_x)
    return distmatrix


def training(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre, digits=4))


def data_set(file):
    mov = ['Data' if i == 0 else 'Data' + str(i) for i in range(18)]
    dreamer = loadmat(file)
    arousal = np.squeeze(dreamer['arousal'])
    valence = np.squeeze(dreamer['valence'])
    x_train = []
    VLC = []
    ARS = []
    PEAR = []
    for num, ch in tqdm(enumerate(mov)):
        data = dreamer[ch][0][0]
        sw = SlidingWindow(size=256, stride=128).fit_transform(data)
        pearson = PearsonDissimilarity().fit_transform(sw)
        G = eegraph.Graph()

        PEAR.append(pearson)
        v = 1 if valence[num] > 3 else 0
        a = 1 if arousal[num] > 3 else 0
        VLC += [v] * len(sw)
        ARS += [a] * len(sw)
    PEAR = np.concatenate(PEAR, axis=0)

    state = np.random.get_state()
    np.random.shuffle(PEAR)
    np.random.set_state(state)
    np.random.shuffle(VLC)
    np.random.shuffle(ARS)
    save_name = '.' + file.split('.')[1]

    distance = PD(PEAR)
    np.savetxt(save_name + '_wst.csv', distance, delimiter=',')
    np.savetxt(save_name + '_vlist.csv', VLC, delimiter=',')
    np.savetxt(save_name + '_alist.csv', ARS, delimiter=',')
    print('save down!')
    return save_name


files = [os.path.join('./dataset', i) for i in os.listdir('./dataset')]

for file in files:
    # name = data_set(file)
    matrix = np.loadtxt('./dataset/s01_wst.csv', delimiter=',')
    print(matrix.shape)

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

    df = pd.read_csv('./dataset/s01_vlist.csv', delimiter=',',header=None)
    df.columns=['class']
    y_ture = df.iloc[testlist]
    y_ture = y_ture.values.tolist()
    y_pred = []
    for i in range(0, len(knn)):
        knnclass = df['class'].iloc[knn[i]]
        totalsum = knnclass.sum()
        average = totalsum/k
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
