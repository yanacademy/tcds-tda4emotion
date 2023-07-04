import numpy as np
from gtda.diagrams import Scaler, PersistenceLandscape
from gtda.pipeline import make_pipeline
from gtda.time_series import SlidingWindow, TakensEmbedding
from gtda.homology import VietorisRipsPersistence
import os


def PD(new_x):
    steps = [
        SlidingWindow(size=128, stride=96),
        TakensEmbedding(time_delay=10, dimension=8),
        VietorisRipsPersistence(),
        Scaler(),
    ]
    tda_pipe = make_pipeline(*steps)
    diagrams = tda_pipe.fit_transform(new_x)
    return diagrams


channel = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
           'O1', 'Oz', 'Pz', 'FP2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
           'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
bands = ['theta', 'alpha', 'beta', 'gamma']
data_dir = '../processed_data'
sid = ['s' + str(i) if i >= 10 else 's0' + str(i) for i in range(1, 33)]
for id in sid:
    file = os.path.join(data_dir, id + '.npy')
    all_data = np.load(file)  # 40*4*32*8064
    all_data = np.swapaxes(all_data, 1, 2)  # 40*32*4*8064

    tmp_trail = []
    for mov in range(40):
        trail_data = all_data[mov]

        tmp_ch = []
        for n_ch, ch in enumerate(channel):
            ch_data = trail_data[n_ch]

            tmp_band = []
            for n_band, band in enumerate(bands):
                diagrams = PD(ch_data[n_band])
                tmp_band.append(diagrams)

            # tmp_band = np.array(tmp_band)
            tmp_ch.append(tmp_band)

        tmp_ch = np.array(tmp_ch)

        tmp_trail.append(tmp_ch)
    # tmp_trail = np.array(tmp_trail)
    np.savez(f'../processed_data/digrams/{id}.npz', tmp_trail)
