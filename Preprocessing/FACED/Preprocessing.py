
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from scipy.signal.windows import hann
from scipy import stats, signal
from mne.viz.topomap import _add_colorbar, plot_topomap, _hide_frame
from mne.preprocessing import ICA
import pickle as pkl
import math

class Preprocessing():

    def __init__(self, raw):
        # Modify the montage
        self.nchns = raw.info['nchan']
        self.freq = raw.info['sfreq']
        old_chn_names = raw.info['ch_names']
        new_chn_names = raw.info['ch_names'].copy()
        if 'A1' in new_chn_names:
            pass
        else:
            new_chn_names[-2] = 'A2'
            new_chn_names[-1] = 'A1'
        chan_names_dict = {old_chn_names[i]: new_chn_names[i] for i in range(32)}
        raw.rename_channels(chan_names_dict)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        # Match the corresponding montage to their index
        self.montage_index = dict(zip(np.arange(self.nchns) , new_chn_names))
        # split out the data matrix
        self.raw = raw
        # Ptr operation
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
        raw_ = self.raw.copy()
        ica.fit(self.raw)
        # Plot different elements of the signals

        # Reject Eog signal
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='Fp1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='Fp2')

        # Reject muscle signal
        muscle_indices,scores = ica.find_bads_muscle(self.raw,threshold=0.91)



        remove_indices = list(set(eog_indices1 + eog_indices2+muscle_indices))


        ica.exclude = remove_indices
        ica.apply(self.raw)

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')

# for mne
def read_data(folderPath):
    # Read the data
    rawdata = mne.io.read_raw_bdf(os.path.join(folderPath, 'data.bdf'), preload=True)
    rawdata,unit = unit_check(rawdata)
    fs = rawdata.info['sfreq']
    # Read the event
    ## read events
    try:
        events = mne.read_annotations(os.path.join(folderPath, 'evt.bdf'))
        onset = np.array([int(x * fs) for x in events.onset] )
        print(onset)
        duration = np.array([int(x) for x in events.duration])
        trigger = np.array([i for i in events.description])
        print("Original trigger events:", trigger)
        # Impedance checking
        trigger,onset,duration,impedance = inter_impedance_inspect(trigger,onset,duration)
        trigger = np.array([int(x) for x in trigger])
        # Check wheter there is ERP data
        trigger,onset, duration, experiment = trigger_check(trigger,onset, duration)
        return trigger, onset, duration, rawdata, [unit, impedance, experiment]
    except:
        raise RuntimeError("No event was found")


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

def eeg_save(subject:str, eegData_trigger:np.array, filepath):

    if len(subject) == 1:
        subject = '00' + subject
    elif len(subject) == 2:
        subject = '0' + subject
    os.makedirs(filepath, exist_ok=True)
    save_path = os.path.join(filepath, f"{subject}.pkl")
    with open(save_path, 'wb') as f:
        pkl.dump(eegData_trigger, f)

# DATA INSPECTION
def unit_check(rawdata):
    # The first batch and the second batch have different unit (uV and V)
    original_raw = rawdata.copy()
    # Here can use np.log to make sure the level of unit, V or uV
    # original_raw._data is a two-dimensional NumPy array of the shape (n_channels, n_samples),
    # where: n_channels is the number of channels in the EEG data. n_samples is the number of samples per channel.
    data_mean = np.mean(np.abs(original_raw._data))
    unit = 'uV'
    if math.log(np.mean(np.abs(original_raw._data))) < 0:
        # print('Unit change :', math.log(data_mean))
        print('Unit change :', data_mean)
        original_raw._data = original_raw._data * 1000 * 1000
        unit = 'V'
    return original_raw, unit

def inter_impedance_inspect(trigger,onset,duration):
    if 'Start Impedance' in trigger:
        pos = np.where((trigger == 'Start Impedance') | (trigger == 'Stop Impedance'))[0]
        # Make sure that there is no impedance before the whole experiment
        trigger_left = trigger[int(pos[0]):]
        # key_trigger_left = np.where((trigger_left!=0)&(trigger_left<29))[0]
        trigger_left = np.delete(trigger_left, [0,1])
        print(trigger_left)
        if trigger_left.size == 0:
            trigger = np.delete(trigger, pos)
            onset = np.delete(onset, pos)
            duration = np.delete(duration, pos)
            print("There is an Impedance in the dataset but at the end")
            impedance = 0
        else:
            print('A impedance occurs in the middle of the data')
            print(trigger)
            raise ValueError('Please Delete This Subject')
    else:
        impedance = 1
        pass
    return trigger, onset, duration, impedance

def trigger_check(trigger,onset,duration):
    trigger_cate = list(set(list(trigger)))
    experiment = 1
    for trigg in [91,92,93,94,95,96]:
        if trigg in trigger_cate:
            experiment = 'ERP'
    if experiment == 'ERP':
        print("There is other dataset inside the data")
        real_Exper = np.where(trigger==100)[0][4]
        real_Exper_end = np.where(trigger==102)[0][-1]+1
        if real_Exper_end > real_Exper:
            pass
        else:
            real_Exper = np.where(trigger == 100)[0][3]
        trigger = trigger[real_Exper:real_Exper_end]
        onset = onset[real_Exper:real_Exper_end]
        duration = duration[real_Exper:real_Exper_end]

    return trigger, onset, duration, experiment


def channel_modify(data, first_or_second):
    # data (33,210000)
    new_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                 30, 31, 17, 16, 32]
    if first_or_second == 1 :
        data = data[np.array(new_order),:]
    # # Delete the reference channel
    # data = np.vstack((data[:-3, :],data[-1,:]))
    chns = 32; fs = 128; n_vids = 28; sec = 30;
    eegdata = np.zeros((n_vids, chns, fs * sec))
    video_index = np.where(data[-1,:].T>0)[0]
    video = data[-1, video_index]
    video_arange = np.argsort(data[-1,video_index])
    video_arange_index = video_index[video_arange]
    # Modify the video order into the normal arrange
    for idx, vid in enumerate(video_arange_index):
        eegdata[idx, :, :] = data[:-1, vid:vid + fs*sec]
    return eegdata