
import mne
import numpy as np
import os
from mne.preprocessing import ICA
import pickle as pkl
import math


class Preprocessing():

    def __init__(self, raw):
        # 存储原始数据中的通道数、采样频、通道名称
        self.nchans = raw.info['nchan']
        self.sfreq = raw.info['sfreq']
        ch_names = raw.info['ch_names']
        # 创建标准的10-20电极布局
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        # 分别存储 EEG 通道和非 EEG 通道的索引和名称的映射关系
        self.montage_index = dict(zip(np.arange(self.nchans), ch_names))
        self.raw = raw
        self.data = self.raw.get_data()


    def band_pass_filter(self,l_freq,h_freq):
        self.raw.filter(l_freq,h_freq)

    def down_sample(self,n_freq):
        self.raw.resample(n_freq)

    # Should cut the data into pieces than do the interpolation
    # def bad_channels_interpolate(self,thresh1=None,thresh2=None,proportion=0.3):
    #     data = self.raw.get_data()
    #     # We found that the data shape of epochs is 3 dims
    #     if len(data.shape) > 2:
    #         data = np.squeeze(data)
    #         #print(data.shape)
    #         #print(data)
    #     Bad_chns = []
    #     value = 0
    #     # Delete the much larger point
    #     if thresh1 != None:
    #         md = np.median(np.abs(data))
    #         value = np.where(np.abs(data) > (thresh1 * md),0,1)[0]
    #     if thresh2 != None:
    #         value = np.where((np.abs(data)) > thresh2,0,1)[0]
    #     # Use the standard to pick out the bad channels
    #     Bad_chns = np.argwhere(np.mean((1-value),axis=1)>proportion)
    #     Bad_chns = Bad_chns.flatten()
    #     # print(type(Bad_chns))
    #     if Bad_chns.size > 0:
    #         self.raw.info['bads'].extend([self.montage_index[bad] for bad in Bad_chns] )
    #         print('Bad channels: ',self.raw.info['bads'])
    #         self.raw = self.raw.interpolate_bads()
    #     else:
    #         print('No bad channel currently')

    def eeg_ica(self,check_ica=None ):
        ica = ICA(random_state=97,max_iter='auto',method='picard')
        ica.fit(self.raw)
        # Plot different elements of the signals

        # Reject Eog signal
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='EXG1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='EXG2')
        eog_indices3, eog_score3 = ica.find_bads_eog(self.raw, ch_name='EXG3')
        eog_indices4, eog_score4 = ica.find_bads_eog(self.raw, ch_name='EXG4')

        remove_indices = list(set(eog_indices1 + eog_indices2+eog_indices3+eog_indices4))

        ica.exclude = remove_indices


        ica.apply(self.raw)
        ica.exclude = []
        self.raw.pick(range(32))
        print(self.raw.info['nchan'])
        ica.fit(self.raw)
        # Reject muscle signal
        muscle_indices,scores = ica.find_bads_muscle(self.raw,threshold=0.91)
        remove_indices = list(set(muscle_indices))
        ica.exclude = remove_indices
        ica.apply(self.raw)

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')



def data_concat(eegData, videoData:np.array, video:int):
    fs = 128; secs = 30
    if len(videoData.shape) > 2:
        videoData = np.squeeze(videoData)
    trigger = np.zeros((1,fs * secs))
    trigger[0][0] = video
    print('The shape of current epoch:', videoData.shape)
    if videoData.shape[1] > fs * secs:
        videoData = videoData[:,-fs*secs:]
    elif videoData.shape[1] < fs * secs:
        raise RuntimeError("The length of epoch is wrong")
    videoData = np.vstack((videoData, trigger))
    if eegData is None:
        eegData = videoData
    else:
        eegData = np.hstack((eegData, videoData))
    return eegData


def unit_check(rawdata):
    # The first batch and the second batch have different unit (uV and V)
    original_raw = rawdata.copy()
    # Here can use np.log to make sure the level of unit, V or uV
    # original_raw._data 是一个二维的 NumPy 数组，形状为 (n_channels, n_samples)，其中：n_channels 是EEG数据中的通道数。n_samples 是每个通道的采样点数。
    data_mean = np.mean(np.abs(original_raw._data[:32,:]))
    unit = 'uV'
    if math.log(data_mean) < 0:
        # print('Unit change :', math.log(data_mean))
        print('Unit change :', data_mean)
        original_raw._data = original_raw._data * 1000 * 1000
        unit = 'V'
    return original_raw, unit
def eeg_save(subject:str, eegData_trigger:np.array, filepath):


    filepath = os.path.join(filepath)  + '/' + subject[:-4] +'.pkl'
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "wb") as f:
        pkl.dump(eegData_trigger,f)
        f.close()


def channel_modify(data, first_or_second):
    # data (33,3840*40),data[-1] contain the number of videos
    new_order = [0,1,3,2,5,4,7,6,9,8,11,10,13,14,15,12,29,28,30,26,27,24,25,31,22,23,20,21,18,19,17,16,32]
    if first_or_second == 1 :
        data = data[np.array(new_order),:]

    chns = 32; fs = 128; n_vids = 40; sec = 30;
    eegdata = np.zeros((n_vids, chns, fs * sec))
    video_index = np.where(data[-1:,:].T>0)[0]
    video = data[-1, video_index]
    video_arange = np.argsort(data[-1,video_index])# get trials sequence
    video_arange_index = video_index[video_arange]
    # Modify the video order into the normal arrange
    for idx, vid in enumerate(video_arange_index):
        eegdata[idx, :, :] = data[:-1, vid:vid + fs*sec]
    return eegdata