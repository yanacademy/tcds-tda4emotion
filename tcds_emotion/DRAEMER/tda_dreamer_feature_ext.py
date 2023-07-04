import os
import pickle
import time

import nolds
import numpy as np

from NLFE.nlfe import poincare_plot, recurrence_plot, approximate_entropy, fuzzy_entropy, sample_entropy
from scipy.signal import butter, filtfilt
from tqdm import tqdm

import scipy.io as sio
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude, NumberOfPoints, PairwiseDistance, PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.pipeline import make_pipeline
from gtda.time_series import TakensEmbedding, SlidingWindow
from scipy.signal import butter, filtfilt
from sklearn.metrics import classification_report


def save_classification_report(result):
    acc = result["accuracy"]
    macro_precision = result["macro avg"]["precision"]
    macro_recall = result["macro avg"]["recall"]
    macro_f1 = result["macro avg"]["f1-score"]
    weighted_precision = result["weighted avg"]["precision"]
    weighted_recall = result["weighted avg"]["recall"]
    weighted_f1 = result["weighted avg"]["f1-score"]


def plot_sub_freq_band(is_plot=False):
    pass


def frequency_band(data, fs=128, rm_fb=None):
    if rm_fb is None:
        rm_fb = ['delta']
    sub_FB = {
        'delta': [np.array([1 * 2 / fs, 4 * 2 / fs]), np.zeros_like(data)],
        'theta': [np.array([4 * 2 / fs, 8 * 2 / fs]), np.zeros_like(data)],
        'alpha': [np.array([8 * 2 / fs, 13 * 2 / fs]), np.zeros_like(data)],
        'beta': [np.array([13 * 2 / fs, 30 * 2 / fs]), np.zeros_like(data)],
        'gamma': [np.array([30 * 2 / fs, 45 * 2 / fs]), np.zeros_like(data)]
    }
    filtered_data = np.zeros((5 - len(rm_fb), data.shape[0], data.shape[1], data.shape[2]))
    selected_sub_band = list(sub_FB.keys())
    for unused_fb in rm_fb:
        selected_sub_band.remove(unused_fb)
    for idx, key in enumerate(selected_sub_band):
        if key in rm_fb:
            continue
        # print(key)
        [k, l] = butter(2, Wn=sub_FB[key][0], btype='bandpass')
        sub_FB[key][1] = filtfilt(k, l, data)
        filtered_data[idx] = sub_FB[key][1]
        # init_data = np.concatenate((init_data, sub_FB[key][1]), axis=0)
    return sub_FB, filtered_data


def sliding_window(data, size=128, stride=96):
    # 样本点为N， 窗口为S ， 步长为W， 则它该样本最终
    # 经过滑动窗口后，得到的窗口样本个数为
    data_len = data.shape[-1]
    win_nums = int((data_len - size) / stride) + 1
    # end_pos = (win_nums - 1) * stride + size
    win_indices = np.zeros((win_nums, 2), dtype=int)
    win_indices[:, 0] = np.arange(data_len)[::stride][:win_nums]
    win_indices[:, 1] = win_indices[:, 0] + size

    return win_indices


def dimension_embeded(data, time_delay=10, dimension=8):
    pass


def extracting_SE(data, win_idx, features):
    # features的地址和传入的参数地址相同
    # print(id(features))
    # 下面的方法就是一个个循环,没有用到向量化计算
    for sub_fb in range(data.shape[0]):
        # print("\n当前处理频带{}".format(sub_fb))
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 下面的m是嵌入的维度，r为1是设置的阈值

                    features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i] = sample_entropy(
                        data_win_i, m=8, r=1)


def extracting_FE(data, win_idx, features):
    for sub_fb in range(data.shape[0]):
        # print("\n当前处理频带{}".format(sub_fb))
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 下面的m是嵌入的维度，n和r都是随机设置的
                    features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i] = fuzzy_entropy(
                        data_win_i, m=8, n=4, r=2)
    # 将各通道的特征合并
    # return features.reshape((features.shape[0], features.shape[1] * features.shape[2]))


def extracting_AE(data, win_idx, features):
    for sub_fb in range(data.shape[0]):
        # print("\n当前处理频带{}".format(sub_fb))
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 下面的m是嵌入的维度，n和r都是随机设置的
                    features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i] = approximate_entropy(
                        data_win_i, m=8, r=1)
    # 将各通道的特征合并
    # return features.reshape((features.shape[0], features.shape[1] * features.shape[2]))


def extracting_RP(data, win_idx, features):
    for sub_fb in range(data.shape[0]):
        # print("\n当前处理频带{}".format(sub_fb))
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 下面的m是嵌入的维度，lag是维度嵌入延迟， theta是阈值
                    _, features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i] = recurrence_plot(
                        data_win_i, m=8, lag=10, theta=1)
    # 将各通道的特征合并


def extracting_PP(data, win_idx, features):
    features_2 = np.copy(features)
    for sub_fb in range(data.shape[0]):
        # print("\n当前处理频带{}".format(sub_fb))
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 返回值依次是SD1, SD2, ratio
                    features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i], features_2[
                        trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i], _ = poincare_plot(
                        data_win_i)
                # 将各通道的特征合并
    # features = np.concatenate((features, features_2), axis=-1)


def extracting_lyap(data, win_idx, features):
    for sub_fb in range(data.shape[0]):
        for trial_i in range(data.shape[1]):
            for win_i in range(win_idx.shape[0]):
                for ch_i in range(data.shape[2]):
                    data_win_i = data[sub_fb, trial_i, ch_i, win_idx[win_i][0]:win_idx[win_i][1]]
                    # 下面的m是嵌入的维度，n和r都是随机设置的
                    features[trial_i * win_idx.shape[0] + win_i, sub_fb * data.shape[2] + ch_i] = nolds.lyap_r(
                        data_win_i,
                        emb_dim=10,
                        lag=0,
                        min_tsep=32,
                        fit="poly")


def draemer2deap(sub_trial_data, data_len, ch_nums=14):
    # 18个 trial 14个channel
    # 3个标签
    # 由于每个trial的中数据的长度是不一致的，因此，这里需要单独对一个trial做转换，提取完特征后再合并为一个人的
    # deap的形状为 (1, 14, x)
    # trials = ['Data' if i == 0 else 'Data' + str(i) for i in range(trial_nums)]
    # labels = ['arousal', 'valence', 'dominance']
    deap_shape_trial_data = np.zeros((1, ch_nums, data_len))
    # deap_shape_label = np.zeros((trial_nums, label_nums), dtype=int)
    # for i in range(label_nums):
    #     deap_shape_label[:, i] = np.squeeze(sub_data[labels[i]])
    # arousal = np.squeeze(sub_data['arousal'])
    # valence = np.squeeze(sub_data['valence'])
    # dominance = np.squeeze(sub_data['dominance'])
    # for tid, trial in enumerate(trials):
    for ch in range(ch_nums):
        deap_shape_trial_data[0, ch] = np.reshape(sub_trial_data[0][0][:, ch], (1, data_len))
    return deap_shape_trial_data


def PD(new_x):
    steps = [
        SlidingWindow(size=win_size, stride=stride),
        TakensEmbedding(time_delay=10, dimension=8),
        VietorisRipsPersistence(),
        Scaler(),
        PersistenceLandscape(n_bins=50)
    ]
    tda_pipe = make_pipeline(*steps)
    temp_rs = tda_pipe.fit_transform(new_x)
    PE = np.mean(temp_rs, axis=1)
    return PE


def extracting_Topo(data, features):
    for trial_i in range(data.shape[1]):
        next_axis1_s = 0
        for ch_i in range(data.shape[2]):
            for sub_fb in range(data.shape[0]):
                data_i = data[sub_fb, trial_i, ch_i]
                axis0_s = trial_i * win_nums
                axis0_e = axis0_s + win_nums
                axis1_s = next_axis1_s
                axis1_e = fea_nums + axis1_s
                features[axis0_s:axis0_e, axis1_s:axis1_e] = PD(data_i)
                next_axis1_s = axis1_e


def remove_nan(data):
    # data[np.isnan(data)] = 0
    # 下面这个函数可以直接将包括nan和inf的值替换成相应的数字，默认分别是0和一个非常大的数字
    # 如果copy参数不为False，那么默认需要重新接收一次返回值
    data = np.nan_to_num(data)
    data[data > 10000] = 10000
    data[data < -10000] = -10000
    return data


if __name__ == '__main__':
    path = "./dataset"
    output_dirs = [
        # 'sampleEn',
        # 'fuzzyEn',
        # 'ApEn',
        # 'RP',
        # 'PP',
        'LYAP'
        # 'TOPO',
    ]
    fea_extracting_method = {
        # "sampleEn": extracting_SE,
        # 'fuzzyEn': extracting_FE,
        # 'ApEn': extracting_AE,
        # 'PP': extracting_PP,
        # 'RP': extracting_RP,
        'LYAP': extracting_lyap,
    }
    win_size = 128
    stride = 96

    """下面的一些设置是针对dreamer数据集的参数"""
    trial_nums = 18
    ch_nums = 14
    label_nums = 3
    sub_fb_nums = 4
    """end"""
    """下面的参数是针对TOPO特征的"""
    win_nums = 0  # 后面赋值
    # 使用时需要解注释
    # fea_nums = 50
    """end"""

    """使用非拓扑特征时要解注释"""
    fea_nums = 1
    #
    have_done_files = []
    for i in range(0, 8):
        if i < 10:
            f_name = 's0{}.mat'.format(i)
        else:
            f_name = 's{}.mat'.format(i)
        have_done_files.append(f_name)
    for output_dir in output_dirs:
        output_path = './features/{}'.format(output_dir)

        for fname in os.listdir(path):
            if fname in have_done_files:
                continue
            print('\n当前处理的文件是{}'.format(fname))
            start_t = time.time()
            fpath = os.path.join(path, fname)
            """下面代码从DRAEMER数据集中读取数据"""
            # raw_labels, concatenated_fea = for_dreamer_dataset()
            sub_data = sio.loadmat(fpath)
            trials = ['Data' if i == 0 else 'Data' + str(i) for i in range(trial_nums)]
            label_list = ['arousal', 'valence', 'dominance']
            raw_labels = np.zeros((trial_nums, label_nums), dtype=int)
            for i in range(label_nums):
                raw_labels[:, i] = np.squeeze(sub_data[label_list[i]])
            concatenated_fea = np.empty((0, sub_fb_nums * ch_nums * fea_nums))
            trial_wins = []
            for trial in tqdm(trials):
                each_trial_len = sub_data[trial][0][0].shape[0]
                raw_data = draemer2deap(sub_data[trial], each_trial_len)
                raw_data = remove_nan(raw_data)
                sub_freq_band_data, concatenated_data = frequency_band(raw_data)
                """下面代码针对拓扑特征"""
                """
                data_len = raw_data.shape[-1]
                win_nums = int((data_len - win_size) / stride) + 1
                fea_nums = 50
                trial_wins.append(win_nums)
                trial_concatenated_fea = np.zeros((concatenated_data.shape[1] * win_nums,
                                                   concatenated_data.shape[0] * concatenated_data.shape[2] * fea_nums))
                extracting_Topo(concatenated_data, trial_concatenated_fea)
                concatenated_fea = np.concatenate((concatenated_fea, trial_concatenated_fea), axis=0)
                """
                """下面代码是针对非拓扑特征的，注意上面的一些参数"""
                # 首先拿到窗口索引
                win_indices = sliding_window(concatenated_data)
                # 记录每个trial的窗口个数，以便于复制标签
                trial_wins.append(win_indices.shape[0])
                # 预分配足够的内存存储计算熵后的空间
                sub_trial_concatenated_fea = np.zeros((concatenated_data.shape[1] * win_indices.shape[0],
                                                       concatenated_data.shape[0] * concatenated_data.shape[2]))
                # 窗口的个数决定了每个trial的每个channel有多少个特征，注意：子频带
                # print(id(concatenated_fea))
                fea_extracting_method[output_dir](concatenated_data, win_indices, sub_trial_concatenated_fea)
                concatenated_fea = np.concatenate((concatenated_fea, sub_trial_concatenated_fea), axis=0)

            """下面的代码是公共部分"""
            raw_labels = raw_labels.repeat(trial_wins, axis=0)
            print("最后特征大小为{}".format(concatenated_fea.shape))
            end_t = time.time()
            print(end_t - start_t)
            out_name = fname.split(".")[0] + ".txt"
            np.savetxt(os.path.join(output_path, out_name),
                       concatenated_fea,
                       delimiter=",")

            labels = np.where(raw_labels > 3, 1, 0)

            label_name = fname.split(".")[0] + "_label.txt"
            np.savetxt(os.path.join(output_path, label_name),
                       labels,
                       fmt='%d',
                       delimiter=",")
