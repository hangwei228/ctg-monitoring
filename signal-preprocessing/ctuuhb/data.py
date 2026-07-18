import numpy as np
import pandas as pd
import os

from config import DATA_DIR


def load_data(split="train"):
    fhr = pd.read_csv(os.path.join(DATA_DIR, f"fhr_{split}.csv"), header=None).values
    uc = pd.read_csv(os.path.join(DATA_DIR, f"uc_{split}.csv"), header=None).values
    fm = pd.read_csv(os.path.join(DATA_DIR, f"fm_{split}.csv"), header=None).values
    y = pd.read_csv(os.path.join(DATA_DIR, f"label_{split}.csv"), header=None).values

    fhr = fhr.reshape(-1, 900, 1).astype(np.float32)
    uc = uc.reshape(-1, 900, 1).astype(np.float32)
    fm = fm.reshape(-1, 900, 1).astype(np.float32)
    y = y.ravel()

    return (fhr, uc, fm), y


def load_sample(fhr_path, uc_path, fm_path):
    fhr = pd.read_csv(fhr_path, header=None).values.reshape(1, 900, 1).astype(np.float32)
    uc = pd.read_csv(uc_path, header=None).values.reshape(1, 900, 1).astype(np.float32)
    fm = pd.read_csv(fm_path, header=None).values.reshape(1, 900, 1).astype(np.float32)
    return [fhr, uc, fm]
