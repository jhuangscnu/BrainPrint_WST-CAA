import os
import numpy as np
import pandas as pd
import scipy.io as sio
from Preprocessing import *
import hdf5storage


if __name__ == '__main__':

    foldPaths = '../../Dataset/FACEDDataset/Recording_info.csv'
    data_dir = '../../Dataset/FACEDDataset/Data'
    save_dir = '../../ProcessedData/FACED'
    save_dir_NoPre = '../../ProcessedData/FACED_NoPre'  # 新增：未经过滤的版本
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_NoPre, exist_ok=True)
    pd_data = pd.read_csv(foldPaths,nrows=123,low_memory=False)
    sub_info = pd_data['sub']
    sub_batch = pd_data['Cohort']
    print(sub_info)

    # Read the data
    for idx, sub in enumerate(sub_info):
   
            
        sub_path = os.path.join(data_dir,sub)
        print("Current processing subject:", sub)
        trigger, onset, duration, rawdata, [unit, impedance, experiments] = read_data(sub_path)

        # read_the_remark_data
        remark_data = hdf5storage.loadmat(os.path.join(sub_path, 'After_remarks.mat'))['After_remark']

        vids = np.squeeze(remark_data['vid'])

        # frequency = rawdata.info['sfreq']

        events = np.transpose(np.vstack((np.vstack((onset, duration)), trigger)))

        # The first batch and the second batch have different unit (uV and V)
        original_raw = rawdata.copy()

        # Epochs cutting
        # Here, the last 30s segment is first captured. During the WST operation, the data will be captured to the 24s segment from the first 3s to the 27s
        cut_seconds = -30; event_id = 102;
        epochs = mne.Epochs(original_raw, events, event_id=event_id, tmin=cut_seconds, tmax=0, preload=True)

        print(type(epochs[0]))

        # Trigger segmentation(trigger->trial，trigger[0]->video_index)
        video_trigger_index = np.where((trigger!=0)&(trigger<29))[0]

        eeg_Data_saved = None
        eeg_Data_saved_raw = None
        #Index:the trials(but start from 0,which in table is 1,Pos: the videos(Unordered,order by experiments)
        for index,pos in enumerate(video_trigger_index):

            video = trigger[pos]

            # The final 30s trial

            processed_epoch_ = Preprocessing(epochs[index])
            processed_epoch_.down_sample(128)
            processed_epoch_.band_pass_filter(0.5, 42)
            #processed_epoch_.bad_channels_interpolate(thresh1=3, proportion=0.3)
            processed_epoch_.eeg_ica()
            processed_epoch_.average_ref()

            # Save the data
            eeg_Data_saved = data_concat(eeg_Data_saved, processed_epoch_.raw.get_data(), video)

            raw_epoch = Preprocessing(epochs[index])
            raw_epoch.raw.pick(range(32))
            raw_epoch.down_sample(128)
            # only down_sample
            eeg_Data_saved_raw = data_concat(eeg_Data_saved_raw, raw_epoch.raw.get_data(), video)

        # Modify the channels
        if int(sub_batch[idx]==1):
            batch = 1
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)
            eeg_Data_saved_raw = channel_modify(eeg_Data_saved_raw, batch)
        elif int(sub_batch[idx]==2):
            batch = 2
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)
            eeg_Data_saved_raw = channel_modify(eeg_Data_saved_raw, batch)

        #Delete HEOR and HEOL channel
        eeg_Data_saved = eeg_Data_saved[:,:-2,:]
        eeg_Data_saved_raw = eeg_Data_saved_raw[:,:-2,:]

        # Saved as pkl
        eeg_save(sub, eeg_Data_saved, save_dir)
        eeg_save(sub, eeg_Data_saved_raw, save_dir_NoPre)
