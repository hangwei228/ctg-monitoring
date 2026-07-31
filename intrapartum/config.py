import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

DATA_ROOT = './dataset_new/'

BATCH_SIZE = 32
EPOCH = 200
MODE = 'AFF'
MODEL_NAME = 'FHR+UC+Clinic+AFF+2025_AFF'
RESULT_DIR = './Result/'
CUDA_VISIBLE_DEVICES = "0"
INPUT_SHAPE_1 = (2, 6000, 1)
INPUT_SHAPE_2 = (1, 14)
OPTIMIZER = 'adam'
LR = 3.5e-5