import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import h5py


def fill_with_last_observation(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1])[:, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None, None], idx, np.arange(idx.shape[2])[None, None, :]]
    out = np.nan_to_num(out)  # if nan still exists then fill with 0
    return out

def fill_with_zeors(arr):
    mask = np.isnan(arr)
    arr[mask] = 0
    return arr

def get_csv_path(data_name):
    if 'electricity' in data_name:
        path = '/data/pdz/incomplete/dataset/electricity/electricity.csv'
    elif 'exchange' in data_name:
        path = '/data/pdz/incomplete/dataset/exchange_rate/exchange_rate.csv'
    elif 'illness' in data_name:
        path = '/data/pdz/incomplete/dataset/illness/national_illness.csv'
    elif 'weather' in data_name:
        path = '/data/pdz/incomplete/dataset/weather/weather.csv'
    elif 'ETTh1' in data_name:
        path = '/data/pdz/incomplete/dataset/ETT-small/ETTh1.csv'
    elif 'ETTh2' in data_name:
        path = '/data/pdz/incomplete/dataset/ETT-small/ETTh2.csv'
    elif 'ETTm1' in data_name:
        path = '/data/pdz/incomplete/dataset/ETT-small/ETTm1.csv'
    elif 'ETTm2' in data_name:
        path = '/data/pdz/incomplete/dataset/ETT-small/ETTm2.csv'
    return path

def get_split(data_name, df_raw, seq_len):
    if 'ETTh1' in data_name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ETTm1' in data_name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif 'ETTh2' in data_name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ETTm2' in data_name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
    return border1s, border2s

def get_scaler(data_name, missing_data, seq_len):
    df_raw = pd.read_csv(get_csv_path(data_name))
    
    cols = list(df_raw.columns)
    cols.remove('OT')
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + ['OT']]
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    # print('missing_data:', missing_data[:100])
    
    border1s, border2s = get_split(data_name, df_raw, seq_len)
    train_data = df_data[border1s[0]: border2s[0]].values
    val_data = df_data[border1s[1]: border2s[1]].values
    test_data = df_data[border1s[2]: border2s[2]].values
    # print(train_data.shape)
    # print(missing_data.shape)
    train_data[np.isnan(missing_data)] = np.nan
    scaler = StandardScaler()
    # print('test:', test_data[96:100])
    train_data = scaler.fit_transform(train_data)
    return scaler
    

class Incomplete_Forecasting_Dataset(Dataset):
    def __init__(self, data, origin_root_path, imputed_root_path, flag, size, 
                impute_method='SAITS'):
        super().__init__()
        
        self.flag = flag
        self.impute_zero = (impute_method == 'ZERO')
        self.impute_method=impute_method
        
        read_origin_data_path = os.path.join(origin_root_path, data + '.h5')
        with h5py.File(read_origin_data_path, "r") as hf:
            self.data = hf[flag]['data'][:]
            self.data_stamp = hf[flag]['data_stamp'][:]
            if self.flag == 'test':
                self.complete_data = hf[flag]['complete_data'][:]
                
            self.train_data = hf['train']['data'][:]
            self.scaler = get_scaler(data, self.train_data,  size[0])
            
        if impute_method != 'ZERO':
            read_imputed_data_path = os.path.join(imputed_root_path,\
                impute_method, data + '_' + impute_method + '_imputations.h5')
            with h5py.File(read_imputed_data_path, "r") as hf:
                self.imputed_x = hf[flag][:]
            assert size[0] == self.imputed_x.shape[1]
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

    def inverse(self, x):
        x = self.scaler.inverse_transform(x)
        return x
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
            
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        origin_x = self.data[s_begin:s_end].copy()
        
        if self.flag == 'test':
            origin_y = self.complete_data[r_begin:r_end].copy()
        else:
            origin_y = self.data[r_begin:r_end].copy()
            
        x_mark = self.data_stamp[s_begin:s_end].copy()
        y_mark = self.data_stamp[r_begin:r_end].copy()
        
        mask_x = (~np.isnan(origin_x)).astype(np.int32)
        mask_y = (~np.isnan(origin_y)).astype(np.int32)

        origin_y = np.nan_to_num(origin_y)
        origin_x = np.nan_to_num(origin_x)
        
        if self.impute_zero:
            return origin_x, origin_y, x_mark, y_mark, mask_x, mask_y
        else:
            return self.imputed_x[index], origin_y, x_mark, y_mark, mask_x, mask_y
    