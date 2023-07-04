import os
import numpy as np
import pickle

from scipy.signal import butter, filtfilt
from tqdm import tqdm
from gtda.time_series import SlidingWindow,TakensEmbedding


def frequency_band(data, fs=128):
    wn = {
        'theta': [4 * 2 / fs, 8 * 2 / fs],
        'alpha': [8 * 2 / fs, 13 * 2 / fs],
        'beta': [13 * 2 / fs, 25 * 2 / fs],
        'gamma': [25 * 2 / fs, 45 * 2 / fs]
    }
    new_data = []
    for key, w in wn.items():
        [k, l] = butter(2, Wn=w, btype='bandpass')
        result = filtfilt(k, l, data)
        new_data.append(result)
    return np.array(new_data)


data_dir = '../dataset'
channel = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
           'O1', 'Oz', 'Pz', 'FP2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
           'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
sid = ['s'+str(i) if i >= 10 else 's0'+str(i) for i in range(1, 33)]
valence = []
arousal = []

for id in sid:
    out = []
    tmp_v = []
    tmp_a = []
    file = os.path.join(data_dir, id+'.dat')
    with open(file, 'rb') as f:
        subject = pickle.load(f, encoding='latin1')
        for i in tqdm(range(0, 40)):  # 40 movie
            data = subject['data'][i]
            labels = subject['labels'][i]
            eeg_data = data[:32]  # 32*8064
            band_data = frequency_band(eeg_data)  # 4*32*8064
            v = 1 if labels[0] > 5 else 0
            a = 1 if labels[1] > 5 else 0

            tmp_v.append(v)
            tmp_a.append(a)

            out.append(band_data)
        out = np.array(out)  # 40*4*32*8064
        np.save(f'../processed_data/{id}.npy', out)
    valence.append(tmp_v)
    arousal.append(tmp_a)
np.save('../processed_data/valence.npy', valence)  # 32*40
np.save('../processed_data/arousal.npy', arousal)
