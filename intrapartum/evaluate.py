import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef
)

import tensorflow as tf
from model import (
    AFFClassifier, SimpleEncoder, AFF1D,
    SoftThresholdingLocal, SoftThresholdingGlobal, SoftThresholdingOutput,
    focal_loss
)
from data import load_data
from config import PROJECT_ROOT, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE
import os

REPORT_PATH = os.path.join(PROJECT_ROOT, "test_results.txt")


def evaluate():
    print("Loading data...")
    (fhr_test, uc_test, fm_test), y_test = load_data("test")

    print(f"Test samples: {len(y_test)}")
    n_normal = np.sum(y_test == 0)
    n_abnormal = np.sum(y_test == 1)
    print(f"  正常(0): {n_normal}, 异常(1): {n_abnormal}")

    print(f"\nLoading model from {MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            'AFFClassifier': AFFClassifier,
            'SimpleEncoder': SimpleEncoder,
            'AFF1D': AFF1D,
            'SoftThresholdingLocal': SoftThresholdingLocal,
            'SoftThresholdingGlobal': SoftThresholdingGlobal,
            'SoftThresholdingOutput': SoftThresholdingOutput,
        },
        compile=False
    )

    y_prob = model.predict([fhr_test, uc_test, fm_test], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['正常(0)', '异常(1)'],
        digits=4
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob[:, 1])
    acc = np.mean(y_pred == y_test)

    lines = []
    lines.append("=" * 40)
    lines.append("  CTG 胎儿健康分类 — 测试集评估报告")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"测试时间: (加载已训练模型)")
    lines.append(f"模型文件: {MODEL_SAVE_PATH}")
    lines.append("")
    lines.append("—" * 30)
    lines.append("  数据集信息")
    lines.append("—" * 30)
    lines.append(f"总样本:     {len(y_test)}")
    lines.append(f"  正常(0):  {n_normal} ({n_normal/len(y_test)*100:.1f}%)")
    lines.append(f"  异常(1):  {n_abnormal} ({n_abnormal/len(y_test)*100:.1f}%)")
    lines.append("")
    lines.append("—" * 30)
    lines.append("  超参数")
    lines.append("—" * 30)
    lines.append(f"Epochs:       {EPOCHS} (含 EarlyStopping)")
    lines.append(f"Batch size:   {BATCH_SIZE}")
    lines.append(f"Learning rate: {LEARNING_RATE}")
    lines.append(f"Loss:         Focal Loss (gamma=2.0, alpha=0.75)")
    lines.append("")
    lines.append("—" * 30)
    lines.append("  混淆矩阵")
    lines.append("—" * 30)
    lines.append(f"{'':>12} {'预测正常':>8} {'预测异常':>8}")
    lines.append(f"{'真实正常':>12} {cm[0,0]:>8} {cm[0,1]:>8}")
    lines.append(f"{'真实异常':>12} {cm[1,0]:>8} {cm[1,1]:>8}")
    lines.append("")
    lines.append("—" * 30)
    lines.append("  分类指标")
    lines.append("—" * 30)
    lines.append(f"Accuracy:  {acc:.4f}")
    lines.append(f"Kappa:     {kappa:.4f}")
    lines.append(f"MCC:       {mcc:.4f}")
    lines.append(f"AUC:       {auc:.4f}")
    lines.append("")
    lines.append(report)

    report_text = "\n".join(lines)

    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n报告已保存至: {REPORT_PATH}")


if __name__ == '__main__':
    evaluate()
