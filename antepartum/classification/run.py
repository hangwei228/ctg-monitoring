# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, roc_curve,
                             cohen_kappa_score, matthews_corrcoef, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import prepare_data
from model import build_model


def compute_metrics(y_true, y_pred, y_pred_proba, num_classes, save_dir=None, prefix=''):
    print(f"\n{'='*60}")
    print(f"\u6a21\u578b\u8bc4\u4f30\u7ed3\u679c {prefix}")
    print(f"{'='*60}")

    acc = accuracy_score(y_true, y_pred)
    print(f"\n\u51c6\u786e\u7387 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")

    present_labels = sorted(set(y_true) | set(y_pred))
    print(f"\n\u5206\u7c7b\u62a5\u544a:")
    if num_classes == 2:
        target_names = ['\u6b63\u5e38', '\u5f02\u5e38']
    else:
        target_names = ['\u6b63\u5e38', '\u53ef\u7591', '\u75c5\u7406\u6027']
    print(classification_report(y_true, y_pred, labels=present_labels,
                                target_names=[target_names[i] for i in present_labels], digits=4))

    print(f"\u6df7\u6dc6\u77e9\u9635:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nCohen's Kappa: {kappa:.4f}")

    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews\u76f8\u5173\u7cfb\u6570: {mcc:.4f}")

    if num_classes == 2:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        print(f"AUC: {auc:.4f}")
        f1_normal = f1_score(y_true, y_pred, average='binary', pos_label=0)
        f1_abnormal = f1_score(y_true, y_pred, average='binary', pos_label=1)
        f1_w = f1_score(y_true, y_pred, average='weighted')
        print(f"F1-Score (\u6b63\u5e38): {f1_normal:.4f}")
        print(f"F1-Score (\u5f02\u5e38): {f1_abnormal:.4f}")
        print(f"F1-Score (weighted): {f1_w:.4f}")
    else:
        all_classes = np.arange(num_classes)
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted', labels=all_classes)
        print(f"AUC (weighted OVR): {auc:.4f}")
        f1_w = f1_score(y_true, y_pred, average='weighted')
        print(f"F1-Score (weighted): {f1_w:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc:.4f})')
        else:
            for i in range(num_classes):
                y_true_bin = (y_true == i).astype(int)
                if y_true_bin.sum() > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba[:, i])
                    auc_i = roc_auc_score(y_true_bin, y_pred_proba[:, i])
                    plt.plot(fpr, tpr, linewidth=2, label=f'{target_names[i]} (AUC={auc_i:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve {prefix}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'roc_curve_{prefix}.png'), dpi=150)
        plt.close()

    return {
        'accuracy': float(acc), 'auc': float(auc), 'f1': float(f1_w),
        'kappa': float(kappa), 'mcc': float(mcc),
        **({'f1_normal': float(f1_normal), 'f1_abnormal': float(f1_abnormal)} if num_classes == 2 else {})
    }


class CTGDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64, shuffle=True, noise_std=0.01, mixup_alpha=0.2, class_weight=None):
        self.x = [x['fhrbs'], x['uc'], x['fm'], x['stats']]
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.class_weight = class_weight
        self.n = len(y)
        self.num_classes = y.shape[1]
        self.indices = np.arange(self.n)
        if class_weight is not None:
            classes = np.argmax(y, axis=1)
            self.sample_weights = np.array([class_weight[c] for c in classes], dtype=np.float32)
        else:
            self.sample_weights = np.ones(self.n, dtype=np.float32)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, int(np.ceil(self.n / self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        bx = [d[batch_idx] for d in self.x]
        by = self.y[batch_idx]

        for i in range(3):
            noise = np.random.normal(0, self.noise_std, bx[i].shape).astype(np.float32)
            bx[i] = bx[i] + noise

        do_mixup = (self.mixup_alpha > 0 and np.random.random() < 0.3)
        if do_mixup:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)
            perm = np.random.permutation(len(batch_idx))
            for i in range(3):
                bx[i] = lam * bx[i] + (1 - lam) * bx[i][perm]
            by = lam * by + (1 - lam) * by[perm]
            bw = lam * self.sample_weights[batch_idx] + (1 - lam) * self.sample_weights[batch_idx[perm]]
        else:
            bw = self.sample_weights[batch_idx]

        return tuple(bx), by, bw

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_and_evaluate(x_train, x_val, x_test, y_train, y_val, y_test,
                       num_classes, dataset_name, result_dir, quick_test=False):
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    stats_dim = x_train['stats'].shape[1]
    model = build_model(num_classes=num_classes, signal_length=1125,
                        stats_dim=stats_dim, l2_reg=1e-5)

    epochs = 5 if quick_test else 100
    batch_size = 64

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.summary()

    save_dir = os.path.join(result_dir, f'{dataset_name}_{"binary" if num_classes==2 else "3class"}')
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'best_model.keras'),
            monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, cooldown=3, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=30,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    classes = np.unique(y_train)
    if len(classes) > 1:
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, cw))
        print(f"\n\u7c7b\u522b\u6743\u91cd: {class_weight}")
    else:
        class_weight = None

    train_gen = CTGDataGenerator(x_train, y_train_cat, batch_size=batch_size,
                                  shuffle=True, noise_std=0.01, mixup_alpha=0.1,
                                  class_weight=class_weight)
    val_inputs = [x_val['fhrbs'], x_val['uc'], x_val['fm'], x_val['stats']]
    test_inputs = [x_test['fhrbs'], x_test['uc'], x_test['fm'], x_test['stats']]

    print(f"\n{'='*60}")
    print(f"\u5f00\u59cb\u8bad\u7ec3 CNN ...")
    print(f"{'='*60}")

    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=(val_inputs, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - t0
    print(f"CNN\u8bad\u7ec3\u5b8c\u6210! \u8017\u65f6: {train_time:.1f}s ({train_time/60:.1f}min)")

    if os.path.exists(os.path.join(save_dir, 'best_model.keras')):
        model = keras.models.load_model(os.path.join(save_dir, 'best_model.keras'))

    y_pred_proba = model.predict(test_inputs, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    task_name = '\u4e8c\u5206\u7c7b' if num_classes == 2 else '\u4e09\u5206\u7c7b'
    metrics = compute_metrics(y_test, y_pred, y_pred_proba, num_classes,
                              save_dir=save_dir, prefix=f'{dataset_name}_{task_name}')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()

    model.save(os.path.join(save_dir, 'final_model.keras'))
    print(f"\u7ed3\u679c\u5df2\u4fdd\u5b58: {save_dir}/")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='\u591a\u6a21\u6001\u80ce\u76d1\u5206\u7c7b\u6a21\u578b - \u6b63\u5219\u5316\u589e\u5f3a\u7248')
    parser.add_argument('--dataset', type=str, default='center', choices=['center', 'mobile', 'all'])
    parser.add_argument('--task', type=str, default='both', choices=['binary', 'three_class', 'both'])
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--quick_test', action='store_true', help='\u5feb\u901f\u6d4b\u8bd5\u6a21\u5f0f (1\u4e2aepoch)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    base_data_dir = os.path.join(os.path.dirname(script_dir), 'Dataset')

    datasets = []
    if args.dataset in ['center', 'all']:
        datasets.append(('center', os.path.join(base_data_dir, 'center')))
    if args.dataset in ['mobile', 'all']:
        datasets.append(('mobile', os.path.join(base_data_dir, 'mobile')))

    tasks = []
    if args.task in ['binary', 'both']:
        tasks.append(('binary', 2))
    if args.task in ['three_class', 'both']:
        tasks.append(('three_class', 3))

    os.makedirs(args.result_dir, exist_ok=True)
    all_results = {}

    for ds_name, ds_path in datasets:
        for task_name, num_classes in tasks:
            print(f"\n{'#'*70}")
            print(f"# \u6570\u636e\u96c6: {ds_name} | \u4efb\u52a1: {task_name} ({num_classes}\u5206\u7c7b)")
            print(f"{'#'*70}\n")

            x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(
                ds_path, num_classes=num_classes, test_size=0.2, val_size=0.15, random_state=42
            )

            metrics = train_and_evaluate(
                x_train, x_val, x_test, y_train, y_val, y_test,
                num_classes=num_classes, dataset_name=ds_name, result_dir=args.result_dir,
                quick_test=args.quick_test
            )

            key = f"{ds_name}_{task_name}"
            all_results[key] = metrics
            print(f"\n\u5b9e\u9a8c {key} \u5b8c\u6210! "
                  f"Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Kappa={metrics['kappa']:.4f}")

    print(f"\n{'#'*70}")
    print(f"# \u6240\u6709\u5b9e\u9a8c\u7ed3\u679c\u6c47\u603b")
    print(f"{'#'*70}")
    print(f"\n{'Dataset':<15} {'Task':<15} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Kappa':<10} {'MCC':<10}")
    print("-" * 80)
    for key, m in all_results.items():
        parts = key.split('_')
        ds, task = parts[0], '_'.join(parts[1:])
        print(f"{ds:<15} {task:<15} {m['accuracy']:<10.4f} {m['auc']:<10.4f} "
              f"{m['f1']:<10.4f} {m['kappa']:<10.4f} {m['mcc']:<10.4f}")

    with open(os.path.join(args.result_dir, 'all_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("\u591a\u6a21\u6001\u80ce\u76d1\u5206\u7c7b\u6a21\u578b(\u6b63\u5219\u5316\u589e\u5f3a\u7248) - \u5b9e\u9a8c\u7ed3\u679c\u6c47\u603b\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Dataset':<15} {'Task':<15} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Kappa':<10} {'MCC':<10}\n")
        f.write("-" * 80 + "\n")
        for key, m in all_results.items():
            parts = key.split('_')
            ds, task = parts[0], '_'.join(parts[1:])
            f.write(f"{ds:<15} {task:<15} {m['accuracy']:<10.4f} {m['auc']:<10.4f} "
                    f"{m['f1']:<10.4f} {m['kappa']:<10.4f} {m['mcc']:<10.4f}\n")

    print(f"\n\u7ed3\u679c\u5df2\u4fdd\u5b58: {args.result_dir}/all_results.txt")


if __name__ == '__main__':
    main()
