import os
import pandas as pd
import numpy as np
from keras.utils import np_utils
from config import DATA_ROOT


class dataset:
    @staticmethod
    def _csv_path(name):
        return os.path.join(DATA_ROOT, name)

    @staticmethod
    def Load_train():
        data_x_train = pd.read_csv(dataset._csv_path('fhr_bs_uc_train.csv'), header=None)
        data_c_train = pd.read_csv(dataset._csv_path('train_c.csv'))
        data_y_train = pd.read_csv(dataset._csv_path('y_train.csv'), header=None)
        data_y_train = np_utils.to_categorical(data_y_train, num_classes=2)
        data_x_train = data_x_train.values.reshape((625, 2, 6000, 1))
        data_c_train = data_c_train.iloc[:, :-1].values.reshape(625, 1, 14)
        return data_x_train, data_c_train, data_y_train

    @staticmethod
    def Load_val():
        data_x_val = pd.read_csv(dataset._csv_path('fhr_bs_uc_val.csv'), header=None)
        data_c_val = pd.read_csv(dataset._csv_path('val_c.csv'))
        data_y_val = pd.read_csv(dataset._csv_path('y_val.csv'), header=None)
        data_y_val = np_utils.to_categorical(data_y_val, num_classes=2)
        data_x_val = data_x_val.values.reshape((157, 2, 6000, 1))
        data_c_val = data_c_val.iloc[:, :-1].values.reshape(157, 1, 14)
        return data_x_val, data_c_val, data_y_val

    @staticmethod
    def Load_test():
        data_x_test = pd.read_csv(dataset._csv_path('fhr_bs_uc_test.csv'), header=None)
        data_c_test = pd.read_csv(dataset._csv_path('test_c.csv'))
        data_y_test = pd.read_csv(dataset._csv_path('y_test.csv'), header=None)
        data_x_test = data_x_test.values.reshape((336, 2, 6000, 1))
        data_c_test = data_c_test.iloc[:, :-1].values.reshape(336, 1, 14)
        return data_x_test, data_c_test, data_y_test

    @staticmethod
    def Load_data():
        data_x_train = pd.read_csv(dataset._csv_path('fhr_bs_uc_train.csv'), header=None)
        data_x_val = pd.read_csv(dataset._csv_path('fhr_bs_uc_val.csv'), header=None)
        data_c_train = pd.read_csv(dataset._csv_path('train_c.csv'))
        data_c_val = pd.read_csv(dataset._csv_path('val_c.csv'))
        x_train = pd.concat([data_x_train, data_x_val], ignore_index=True)
        c_train = pd.concat([data_c_train, data_c_val], ignore_index=True)
        data_y_train = pd.read_csv(dataset._csv_path('y_train.csv'), header=None)
        data_y_val = pd.read_csv(dataset._csv_path('y_val.csv'), header=None)
        y_train = pd.concat([data_y_train, data_y_val], ignore_index=True)
        x_train = x_train.values.reshape((782, 2, 6000, 1))
        c_train = c_train.iloc[:, :-1].values.reshape(782, 1, 14)
        return x_train, c_train, y_train