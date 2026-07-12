import os
import csv
import pandas as pd

dirpath = './'
savpath = './output/'

os.makedirs(savpath, exist_ok=True)

#input
fhr_df = pd.read_csv(os.path.join(dirpath, 'fhr.csv'), header=None)
uc_df  = pd.read_csv(os.path.join(dirpath, 'uc.csv'),  header=None)
fm_df  = pd.read_csv(os.path.join(dirpath, 'fm.csv'),  header=None)
bs_df  = pd.read_csv(os.path.join(dirpath, 'fhrbs.csv'), header=None)
label_df = pd.read_csv(os.path.join(dirpath, 'label.csv'), header=None)

n_samples = len(fhr_df)
fhrs, ucs, fms, bss = [], [], [], []
fhrs1, ucs1, fms1, bss1 = [], [], [], []
fhrs2, ucs2, fms2, bss2 = [], [], [], []
fhrs3, ucs3, fms3, bss3 = [], [], [], []
label, label1, label2, label3 = [], [], [], []
tot, tot2 = [], []
cou = 0

#baseline
for i in range(n_samples):
    fhr_arr = fhr_df.iloc[i].values.astype(float)
    uc_arr  = uc_df.iloc[i].values.astype(float)
    fm_arr  = fm_df.iloc[i].values.astype(float)
    bs_arr  = bs_df.iloc[i].values.astype(float)
    lbl     = int(label_df.iloc[i].values[0])

    fhr = fhr_arr.tolist()
    uc  = uc_arr.tolist()
    fm  = fm_arr.tolist()
    bs  = (fhr_arr - bs_arr).tolist()

    cou += 1
    fhrs.append(fhr)
    label.append(lbl)
    tot.append([fhr, uc, fm, bs])
    tot2.append([fhr, uc, fm, bs, lbl])
print(cou)
#count = 0
#for i in range(len(fhrs)):
#    for j in range(len(fhrs)):
#        if fhrs[i] == fhrs[j] and i != j:
#            print(i, j)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(tot, label, train_size=0.7, random_state=42, shuffle=True, stratify=label)

#count = 0
#for i in range(len(x_train)):
#    for j in range(len(tot)):
#        if x_train[i][0] == tot2[j][0]:
#            if y_train[i] != tot2[j][4]:
#                print('giao! 第' + str(i) + '个数据出错')
#            else:
#                count += 1
#print(count, len(x_train))
print(len(x_test))
for i in range(len(x_test)):
    fhrs2.append(x_test[i][0])
    ucs2.append(x_test[i][1])
    fms2.append(x_test[i][2])
    bss2.append(x_test[i][3])
    label2.append(y_test[i])


x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=12, shuffle=True, stratify=y_train)
print(len(x_train1), len(x_val))
for i in range(len(x_train1)):
    fhrs1.append(x_train1[i][0])
    ucs1.append(x_train1[i][1])
    fms1.append(x_train1[i][2])
    bss1.append(x_train1[i][3])
    label1.append(y_train1[i])
for i in range(len(x_val)):
    fhrs3.append(x_val[i][0])
    ucs3.append(x_val[i][1])
    fms3.append(x_val[i][2])
    bss3.append(x_val[i][3])
    label3.append(y_val[i])


pd.DataFrame(fhrs1).to_csv(savpath + 'A_fhr_train.csv', header = None, index = False)
pd.DataFrame(ucs1).to_csv(savpath + 'A_uc_train.csv', header = None, index = False)
pd.DataFrame(fms1).to_csv(savpath + 'A_fm_train.csv', header = None, index = False)
pd.DataFrame(bss1).to_csv(savpath + 'A_fhrbs_train.csv', header = None, index = False)
pd.DataFrame(label1).to_csv(savpath + 'A_label_train.csv', header = None, index = False)

pd.DataFrame(fhrs2).to_csv(savpath + 'A_fhr_test.csv', header = None, index = False)
pd.DataFrame(ucs2).to_csv(savpath + 'A_uc_test.csv', header = None, index = False)
pd.DataFrame(fms2).to_csv(savpath + 'A_fm_test.csv', header = None, index = False)
pd.DataFrame(bss2).to_csv(savpath + 'A_fhrbs_test.csv', header = None, index = False)
pd.DataFrame(label2).to_csv(savpath + 'A_label_test.csv', header = None, index = False)

pd.DataFrame(fhrs3).to_csv(savpath + 'A_fhr_val.csv', header = None, index = False)
pd.DataFrame(ucs3).to_csv(savpath + 'A_uc_val.csv', header = None, index = False)
pd.DataFrame(fms3).to_csv(savpath + 'A_fm_val.csv', header = None, index = False)
pd.DataFrame(bss3).to_csv(savpath + 'A_fhrbs_val.csv', header = None, index = False)
pd.DataFrame(label3).to_csv(savpath + 'A_label_val.csv', header = None, index = False)