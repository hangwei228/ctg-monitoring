import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef)

from model import AFFClassifier, focal_loss
from data import load_data
from config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE_EARLY_STOP, PATIENCE_LR, LR_FACTOR, MIN_LR


def train():
    print("Loading data...")
    (fhr_train, uc_train, fm_train), y_train = load_data("train")
    (fhr_val, uc_val, fm_val), y_val = load_data("val")
    (fhr_test, uc_test, fm_test), y_test = load_data("test")

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"FHR shape: {fhr_train.shape}, UC shape: {uc_train.shape}, FM shape: {fm_train.shape}")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

    model = AFFClassifier()
    model.compile(
        loss=focal_loss(gamma=2.0, alpha=0.75),
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['binary_accuracy']
    )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=PATIENCE_LR, min_lr=MIN_LR, verbose=1),
        EarlyStopping(monitor="val_loss", patience=PATIENCE_EARLY_STOP, verbose=1, mode='min'),
    ]

    print("Training...")
    history = model.fit(
        x=[fhr_train, uc_train, fm_train],
        y=y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([fhr_val, uc_val, fm_val], y_val_cat),
        callbacks=callbacks,
        verbose=1
    )

    print("\n========== Test Set Evaluation ==========")
    y_prob = model.predict([fhr_test, uc_test, fm_test], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"测试集样本分布: 正常={np.sum(y_test==0)}, 异常={np.sum(y_test==1)}")
    print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['正常(0)', '异常(1)'])}")
    print(f"Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"MCC:   {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"AUC:   {roc_auc_score(y_test, y_prob[:, 1]):.4f}")

    return model, history


if __name__ == '__main__':
    train()
