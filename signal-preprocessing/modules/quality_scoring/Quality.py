import numpy as np
import pandas as pd
import math
import os
      
def list_files(dir):
    file_list = os.listdir(dir)
    return file_list

def get_data(fhr_df, uc_df, idx):
    fhr = fhr_df.iloc[idx].values.astype(float)
    uc  = uc_df.iloc[idx].values.astype(float)
    return fhr, uc
#Extract missing segments from UC signal
def get_Ym(uc_signal):
    uc_missing = []
    uc_missing_start = -1
    uc_missing_end = -1

    for i, value in enumerate(uc_signal):
        if i > 0 and abs(value - uc_signal[i-1]) <= 1:
            if uc_missing_start == -1:
                uc_missing_start = i - 1
            uc_missing_end = i
        else:
            if uc_missing_start != -1:
                uc_missing.append((uc_missing_start, uc_missing_end))
                uc_missing_start = -1
                uc_missing_end = -1
    
    if uc_missing_start != -1:
        uc_missing.append((uc_missing_start, uc_missing_end))

    uc_missing_lengths = [end - start + 1 for start, end in uc_missing]
    Ym = [l for l in uc_missing_lengths if l > 15]
    return Ym

#Extract missing segments from FHR signal
def get_Xm(fhr_signal):
    fhr_missing = []
    fhr_missing_start = -1
    fhr_missing_end = -1

    for i, value in enumerate(fhr_signal):
        if (value < 50 or value > 200) and i > 0:
            if fhr_missing_start == -1:
                fhr_missing_start = i - 1
            fhr_missing_end = i
        else:
            if fhr_missing_start != -1:
                fhr_missing.append((fhr_missing_start, fhr_missing_end))
                fhr_missing_start = -1
                fhr_missing_end = -1
    if fhr_missing_start != -1:
        fhr_missing.append((fhr_missing_start, fhr_missing_end))

    fhr_missing_lengths = [end - start + 1 for start, end in fhr_missing]
    Xn = [l for l in fhr_missing_lengths if l > 0]
    return Xn

#Compute signal quality Q ∈ [0,1],higher is better
def get_Q(fhr_signal, uc_signal):
    l = len(uc_signal)    
    Y = get_Ym(uc_signal)
    X = get_Xm(fhr_signal)
    Xn_log_Xn_sum = sum([(Xn * math.log10(Xn)) for Xn in X])
    Ym_log_Ym_sum = sum([(Ym * math.log10(Ym)) for Ym in Y])
    Q = 1 - ((Xn_log_Xn_sum + Ym_log_Ym_sum) / (2 * l * math.log10(l)))
    return Q
 

if __name__ == '__main__':
    fhr_df = pd.read_csv('./fhr.csv', header=None)
    uc_df  = pd.read_csv('./uc.csv',  header=None)
    n = len(fhr_df)
    print(f'共 {n} 个样本')
    Q_all, index = [], []
    print('Begin...')
    for i in range(n):
        print(f'{i}.csv')
        fhr, uc = get_data(fhr_df, uc_df, i)
        q = get_Q(fhr, uc)  
        Q_all.append(q)
        if q < 0.6:  
            index.append(i)
    print('finish!')
    Q = [i for i in Q_all if i < 0.6]  
    print(index)




