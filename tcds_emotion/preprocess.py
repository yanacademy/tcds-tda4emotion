import os
import numpy as np
from gtda.curves import StandardFeatures
from tqdm import tqdm
import mne
from scipy import signal
from scipy.io import loadmat
from gtda.diagrams import Scaler, PersistenceEntropy, BettiCurve, PersistenceLandscape, PersistenceImage
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding, SlidingWindow
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_diagrams(x):
    embedding_dimension = 8
    embedding_time_delay = 10
    embedder = TakensEmbedding(time_delay=embedding_time_delay, dimension=embedding_dimension)
    persistence = VietorisRipsPersistence()
    scaler = Scaler()

    steps = [('embedder', embedder),
             ('persistence', persistence),
             ('scaler', scaler)]
    pipe = Pipeline(steps)
    diagrams = pipe.fit_transform(x)
    return diagrams


def get_sbj_file(data_dir):
    """
    :param data_dir: eeg file path
    :return: list for each subject [[sbj1],[sbj2]]
    """
    file = [i for i in os.listdir(data_dir) if i[-4:] == '.mat']
    file.sort(key=lambda x: int(x.split('_')[0]))
    sbj_file = []
    start = 0
    for i in range(len(file)-1):
        if file[i].split('_')[0] != file[i+1].split('_')[0]:
            sbj_file.append(file[start:i+1])
            start = i+1
    return sbj_file


def channel_select(ch_data, ch, is_filter=True):
    """
    :param ch_data: the data that has extracted for each trail about each subject
    :param ch: select channel list
    :param is_filter: whether adopt filter to 1-50Hz
    :return: filtered array or selected channel array
    """
    info = mne.create_info(channel, sfreq=200, ch_types='eeg')
    raw = mne.io.RawArray(ch_data, info=info)
    raw_selected = raw.pick_channels(ch)
    if is_filter:
        filtered = raw_selected.filter(1, 50, fir_design='firwin').to_data_frame(index='time')
        return np.array(filtered)
    else:
        return np.array(raw_selected.to_data_frame(index='time'))


def clear_dict(file_name):
    """
    :param file_name: want to read file name
    :return: all channel data of all trail eg:15_trail * 62_channel * n_signal
    """
    eeg_dir = './Preprocessed_EEG/'
    data = loadmat(os.path.join(eeg_dir, file_name))
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    return data.values()


channel = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
            'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
            'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
select_ch = ['FT7', 'T7', 'TP7', 'P7', 'C5', 'CP5', 'FT8', 'T8', 'TP8', 'P8', 'C6', 'CP6']

label_dir = './label.mat'
labels = np.squeeze(loadmat(label_dir)['label'])

eeg_dir = './Preprocessed_EEG/'
for sid, idx in enumerate(get_sbj_file(eeg_dir)):  # 获取每个subject的文件列表
    print(f'第{sid+1} subject:')
    for file_name in idx:  # 读取subject的每个eeg文件
        tmp = []
        y = []
        for i, trail in enumerate(clear_dict(file_name)):  # 获取每个测试文件的15个电影测试eeg，并对每个电影eeg处理
            x = channel_select(trail, ch=channel, is_filter=False)
            x = SlidingWindow(size=1000, stride=1000).fit_transform(x).transpose(0, 2, 1)
            diagrams = get_diagrams(x)
            pe = PersistenceEntropy(normalize=True).fit_transform(diagrams)
            bc = BettiCurve(n_bins=50).fit_transform(diagrams)
            bc = np.mean(bc, axis=1)
            pl = PersistenceLandscape(n_bins=50).fit_transform(diagrams)
            pl = pl.sum(axis=1)
            tmp.append(bc)
            y.extend([labels[i]]*len(bc))

        feature = np.concatenate(tmp, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(feature, y, shuffle=True, test_size=0.3)
        model = RandomForestClassifier(random_state=1)
        model.fit(x_train, y_train)
        print(classification_report(y_test, model.predict(x_test)))



