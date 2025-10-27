import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import pandas as pd

import scipy.io as sio
from Preprocessing import *



if __name__ == '__main__':

    foldPaths = '../../Dataset/DEAPDataset/participant_ratings.xls'
    data_dir = '../../Dataset/DEAPDataset/DEAP_Raw'
    save_dir = '../../ProcessedData/DEAP'
    save_dir_NoPre = '../../ProcessedData/DEAP_NoPre'  # 新增：未经过滤的版本
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_NoPre, exist_ok=True)

    #s24,s25
    sub_info =  [f's{i:02d}.bdf' for i in range(1, 33)]
    sub_batch = [1] * 22 + [2] * 10

    print(sub_info)

    # Read the data
    for idx, sub in enumerate(sub_info):
        sub_path = os.path.join(data_dir,sub)
        print("Current processing subject:", sub)
        rawdata = mne.io.read_raw_bdf(sub_path, preload=True)

        if 23 <= idx <= 27:
            rename_dict = {'': 'Status'}
            rawdata.rename_channels(rename_dict)
        elif 28 <= idx <= 31:
            rename_dict = {'-1': 'Status'}
            rawdata.rename_channels(rename_dict)
        print(rawdata.info['ch_names'])
        events = mne.find_events(rawdata,stim_channel='Status')


        #get_video_list
        df = pd.read_excel(foldPaths, engine='xlrd')
        start_row = idx*40
        end_row = start_row+40
        vids = df.iloc[start_row:end_row,2].tolist()
        print(vids)

        #convert all data to consistent units
        rawdata, unit = unit_check(rawdata)
        original_raw = rawdata.copy()
        print(original_raw.info)


        # Epochs cutting
        # sub1-23,event_id=5 which is the end of video,sub24-32,event_id = 1638149 which is the end of video
        # but there are 2 extra events at the start of signal in sub23-sub32,we need to exclude it.
        # Here, the last 30s segment is first captured. During the WST operation, the data will be captured to the 24s segment from the first 3s to the 27s
        if idx<=22:
            cut_seconds = -30;event_id = 5
        elif idx==27:
            cut_seconds = -30
            event_id = 1638149
            event_id =[event_id,5832452]
        else :
            cut_seconds = -30; event_id = 1638149

        epochs = mne.Epochs(original_raw, events, event_id=event_id, tmin=cut_seconds, tmax=0, preload=True)
        if idx>=22:
            epochs = epochs[2:]
        print(epochs.__len__())



        eeg_Data_saved = None
        eeg_Data_saved_raw = None
        #Index:the trials(but start from 0,which in table is 1,Pos: the videos(Unordered,order by experiments)
        for index in range(40):

            video = vids[index]

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


        # Saved as pkl

        eeg_save(sub, eeg_Data_saved, save_dir)
        eeg_save(sub, eeg_Data_saved_raw, save_dir_NoPre)

