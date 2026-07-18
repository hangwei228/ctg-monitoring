import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "..", "..", "..",
    "筛选_去噪结果_ctuuhb(1)",
    "筛选_去噪结果_ctuuhb",
    "ctuuhb_result_2",
)

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "..", "best_model.keras")

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 3.5e-5
PATIENCE_EARLY_STOP = 15
PATIENCE_LR = 3
LR_FACTOR = 0.92
MIN_LR = 1e-7
