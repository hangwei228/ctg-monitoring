import copy
import tensorflow as tf
import numpy as np
from keras.utils import Sequence


def _apply_block_mask(x, seq_length, num_blocks_range=(1, 10), block_ratio_range=(0.02, 0.1)):
    num_blocks = tf.random.uniform([], num_blocks_range[0], num_blocks_range[1] + 1, dtype=tf.int32)
    block_ratio = tf.random.uniform([], block_ratio_range[0], block_ratio_range[1])
    block_len = tf.cast(seq_length * block_ratio, tf.int32)
    block_len = tf.maximum(block_len, 1)
    for _ in range(num_blocks):
        start = tf.random.uniform([], minval=0, maxval=seq_length - block_len, dtype=tf.int32)
        x[..., start:start + block_len] = 0.0


def _apply_c_block_mask(c, c_seq_length):
    block_len = tf.random.uniform([], 1, 3, dtype=tf.int32)
    start = tf.random.uniform([], minval=0, maxval=c_seq_length - block_len, dtype=tf.int32)
    c[..., start:start + block_len] = 0.0


class MaskedDataGenerator(Sequence):
    def __init__(self, data_x, data_c, data_y, batch_size):
        self.data_x = data_x
        self.data_c = data_c
        self.data_y = data_y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data_x) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.data_x))
        batch_x = self.data_x[start:end]
        batch_c = self.data_c[start:end]
        batch_y = self.data_y[start:end]
        fhr = batch_x[:, 0:1, :, :]
        uc = batch_x[:, 1:2, :, :]
        batch_fhr1 = copy.deepcopy(fhr)
        batch_fhr2 = copy.deepcopy(fhr)
        batch_uc1 = copy.deepcopy(uc)
        batch_uc2 = copy.deepcopy(uc)
        seq_length = fhr.shape[2]
        c_seq_length = batch_c.shape[2]
        for idx in range(end - start):
            _apply_block_mask(batch_fhr1[idx], seq_length)
            _apply_block_mask(batch_uc1[idx], seq_length)
            _apply_block_mask(batch_fhr2[idx], seq_length)
            _apply_block_mask(batch_uc2[idx], seq_length)
            _apply_c_block_mask(batch_c[idx], c_seq_length)
        return [batch_fhr1, batch_fhr2, batch_uc1, batch_uc2, batch_c], batch_y

    def on_epoch_end(self):
        idx = np.arange(len(self.data_x))
        np.random.shuffle(idx)
        self.data_x = self.data_x[idx]
        self.data_c = self.data_c[idx]
        self.data_y = self.data_y[idx]


class MaskedTestDataGenerator(Sequence):
    def __init__(self, data_x, data_c, data_y, batch_size):
        self.data_x = data_x
        self.data_c = data_c
        self.data_y = data_y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data_x) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.data_x))
        batch_x = self.data_x[start:end]
        batch_c = self.data_c[start:end]
        batch_y = self.data_y[start:end]
        fhr = batch_x[:, 0:1, :, :]
        uc = batch_x[:, 1:2, :, :]
        return [fhr, uc, batch_c], batch_y


class MockedEvaluteDataGenerator(Sequence):
    def __init__(self, data_x, data_c, data_y, batch_size):
        self.data_x = data_x
        self.data_c = data_c
        self.data_y = data_y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data_x) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.data_x))
        batch_x = self.data_x[start:end]
        batch_c = self.data_c[start:end]
        batch_y = self.data_y[start:end]
        fhr = batch_x[:, 0:1, :, :]
        uc = batch_x[:, 1:2, :, :]
        return fhr, uc, batch_c