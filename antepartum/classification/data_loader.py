# -*- coding: utf-8 -*-
"""
数据加载与预处理模块 - 混合归一化版
基于胎监指南提取临床特征
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(data_dir):
    """加载原始数据"""
    fhrbs = pd.read_csv(os.path.join(data_dir, 'fhrbs.csv'), header=None).values.astype(np.float32)
    fm = pd.read_csv(os.path.join(data_dir, 'fm.csv'), header=None).values.astype(np.float32)
    uc = pd.read_csv(os.path.join(data_dir, 'uc.csv'), header=None).values.astype(np.float32)
    gest_age = pd.read_csv(os.path.join(data_dir, 'gest+age.csv'), header=None).values.astype(np.float32)
    labels = pd.read_csv(os.path.join(data_dir, 'label.csv'), header=None).values.astype(np.int32).flatten()
    gest = gest_age[:, 0].reshape(-1, 1)
    age = gest_age[:, 1].reshape(-1, 1)
    return fhrbs, fm, uc, gest, age, labels


def create_three_class_labels(fhrbs, labels):
    """基于FHR信号变异性 + 胎监指南特征 创建三分类标签"""
    new_labels = labels.copy()
    abnormal_idx = (labels == 1)
    fhr_abnormal = fhrbs[abnormal_idx]
    fhr_vars = np.var(fhr_abnormal, axis=1)
    fhr_stds = np.std(fhr_abnormal, axis=1)

    baseline_vals = np.mean(fhr_abnormal[:, 500:], axis=1)

    abnormal_indices = np.where(abnormal_idx)[0]
    for i, idx in enumerate(abnormal_indices):
        bl = baseline_vals[i]
        fv = fhr_vars[i]
        fs = fhr_stds[i]

        is_pathological = (
            bl < 100 or bl > 180 or
            fs < 3 or (fs < 5 and fv < 25) or
            fv > 50
        )

        if is_pathological:
            new_labels[idx] = 2
        else:
            new_labels[idx] = 1

    return new_labels


def compute_fhr_baseline(fhr_signal, fs=1.25):
    """计算FHR基线 (基于胎监指南: 10min内振幅稳定在5bpm以内的均值)"""
    smoothed = gaussian_filter1d(fhr_signal, sigma=2)
    window = int(10 * fs)
    if len(smoothed) < window:
        return np.mean(fhr_signal)
    baselines = []
    for i in range(0, len(smoothed), window):
        seg = smoothed[i:i + window]
        if len(seg) < window // 2:
            continue
        m = np.mean(seg)
        stable = seg[np.abs(seg - m) <= 5]
        if len(stable) > len(seg) * 0.5:
            baselines.append(np.mean(stable))
        else:
            baselines.append(m)
    if len(baselines) == 0:
        return np.mean(fhr_signal)
    return np.median(baselines)


def extract_ctg_features(fhr_signal, uc_signal, fm_signal, fs=1.25):
    """基于胎监指南提取临床特征"""
    n = len(fhr_signal)
    features = {}

    baseline = compute_fhr_baseline(fhr_signal)
    features['baseline'] = baseline

    detrended = fhr_signal - baseline
    features['variability_std'] = np.std(detrended[detrended > -30])
    features['variability_mean'] = np.mean(np.abs(detrended))

    diffs = np.abs(np.diff(detrended))
    features['stv'] = np.mean(diffs)

    zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)
    features['variability_freq'] = zero_crossings / n

    accelerations = 0
    decelerations = 0
    early_dec = 0
    late_dec = 0
    variable_dec = 0

    is_accel = False
    accel_start = -1
    accel_peak = 0
    min_sep = int(15 * fs)

    i = 0
    while i < n - min_sep:
        if detrended[i] > 15:
            peak_val = detrended[i]
            peak_idx = i
            j = i
            while j < n and detrended[j] > 15:
                if detrended[j] > peak_val:
                    peak_val = detrended[j]
                    peak_idx = j
                j += 1
            duration = j - i
            if duration >= min_sep and peak_val >= 15:
                accelerations += 1
            i = j
        else:
            i += 1

    i = 0
    min_dec_dur = int(15 * fs)
    while i < n - min_dec_dur:
        if detrended[i] < -15:
            trough_val = detrended[i]
            trough_idx = i
            j = i
            while j < n and detrended[j] < -15:
                if detrended[j] < trough_val:
                    trough_val = detrended[j]
                    trough_idx = j
                j += 1
            duration = j - i
            if duration >= min_dec_dur and trough_val <= -15:
                decelerations += 1
                is_late = False
                if trough_idx > int(10 * fs) and len(uc_signal) > trough_idx:
                    uc_around = uc_signal[max(0, trough_idx - int(30 * fs)):min(n, trough_idx + int(30 * fs))]
                    if len(uc_around) > 0:
                        uc_peak = np.max(uc_around)
                        if uc_peak > np.median(uc_signal) + np.std(uc_signal):
                            delay = 0
                            for k in range(max(0, trough_idx - int(15 * fs)), min(n, trough_idx + int(15 * fs))):
                                if uc_signal[k] > 0.8 * uc_peak and k > trough_idx:
                                    delay = (k - trough_idx) / fs
                                    break
                            if delay > 10:
                                is_late = True

                if is_late:
                    late_dec += 1
                elif trough_val < -60:
                    variable_dec += 1
                else:
                    early_dec += 1
            i = j
        else:
            i += 1

    features['acceleration_count'] = accelerations
    features['deceleration_count'] = decelerations
    features['early_deceleration'] = early_dec
    features['late_deceleration'] = late_dec
    features['variable_deceleration'] = variable_dec

    decel_area = 0
    depths = []
    i = 0
    while i < n:
        if detrended[i] < -15:
            j = i
            seg_depths = []
            while j < n and detrended[j] < -15:
                seg_depths.append(abs(detrended[j]))
                j += 1
            if len(seg_depths) > min_dec_dur:
                depths.extend(seg_depths)
                decel_area += np.sum(seg_depths)
            i = j
        else:
            i += 1
    features['deceleration_depth'] = np.mean(depths) if depths else 0
    features['deceleration_area'] = decel_area / n if n > 0 else 0

    n_small = max(1, n // 20)
    segments_std = [np.std(detrended[k:k + n_small]) for k in range(0, n, n_small)]
    features['min_epoch_std'] = np.min(segments_std)
    features['max_epoch_std'] = np.max(segments_std)

    uc_std = np.std(uc_signal)
    uc_mean = np.mean(uc_signal)
    uc_peaks = 0
    for k in range(1, n - 1):
        if uc_signal[k] > uc_signal[k - 1] and uc_signal[k] > uc_signal[k + 1] and uc_signal[k] > uc_mean + uc_std:
            uc_peaks += 1
    features['uc_contraction_count'] = uc_peaks

    fm_total = np.sum(fm_signal)
    fm_positive = np.sum(fm_signal > 0)
    features['fm_total'] = fm_total
    features['fm_count'] = fm_positive

    lf = np.mean(np.abs(np.diff(detrended)))
    features['short_term_variation'] = lf

    long_win = int(60 * fs)
    if n > long_win:
        long_segments = [np.std(detrended[k:k + long_win]) for k in range(0, n, long_win)]
        features['long_term_variation'] = np.mean(long_segments)
    else:
        features['long_term_variation'] = np.std(detrended)

    freqs, psd = signal.welch(detrended, fs=fs, nperseg=min(256, n))
    low_band = (freqs >= 0.003) & (freqs <= 0.05)
    high_band = (freqs >= 0.05) & (freqs <= 0.5)
    features['lf_power'] = np.sum(psd[low_band]) if np.any(low_band) else 0
    features['hf_power'] = np.sum(psd[high_band]) if np.any(high_band) else 0
    features['lf_hf_ratio'] = (features['lf_power'] / (features['hf_power'] + 1e-8))

    return features


def extract_batch_ctg_features(fhrbs, uc, fm):
    """批量提取CTG临床特征"""
    n = len(fhrbs)
    feat_list = []
    for i in range(n):
        feat = extract_ctg_features(fhrbs[i], uc[i], fm[i])
        feat_list.append(feat)
    df = pd.DataFrame(feat_list)
    return df.values.astype(np.float32)


def per_sample_normalize(signals):
    """逐样本z-score归一化"""
    means = np.mean(signals, axis=1, keepdims=True)
    stds = np.std(signals, axis=1, keepdims=True) + 1e-8
    return (signals - means) / stds


def extract_signal_stats(signals):
    """提取基本统计特征"""
    n = len(signals)
    stats = np.zeros((n, 7), dtype=np.float32)
    stats[:, 0] = np.mean(signals, axis=1)
    stats[:, 1] = np.std(signals, axis=1)
    stats[:, 2] = np.min(signals, axis=1)
    stats[:, 3] = np.max(signals, axis=1)
    stats[:, 4] = stats[:, 3] - stats[:, 2]
    stats[:, 5] = np.median(signals, axis=1)
    stats[:, 6] = (stats[:, 0] - stats[:, 5]) / (stats[:, 1] + 1e-8)
    return stats


def prepare_data(data_dir, num_classes=2, test_size=0.2, val_size=0.15, random_state=42):
    """完整数据准备 - 混合归一化 + CTG临床特征 (带缓存)"""
    fhrbs, fm, uc, gest, age, labels = load_raw_data(data_dir)

    print(f"数据集: {data_dir}")
    print(f"总样本数: {len(labels)}")
    print(f"原始标签分布: {np.bincount(labels)}")

    if num_classes == 3:
        labels = create_three_class_labels(fhrbs, labels)
        print(f"三分类标签分布: {np.bincount(labels, minlength=3)}")

    fhr_stats = extract_signal_stats(fhrbs)
    fm_stats = extract_signal_stats(fm)
    uc_stats = extract_signal_stats(uc)

    cache_path = os.path.join(data_dir, 'ctg_features_cache.npy')
    if os.path.exists(cache_path):
        print(f"加载缓存的CTG特征: {cache_path}")
        ctg_features = np.load(cache_path).astype(np.float32)
    else:
        print("提取CTG临床特征 (首次运行, 将缓存结果)...")
        ctg_features = extract_batch_ctg_features(fhrbs, uc, fm)
        np.save(cache_path, ctg_features)
        print(f"CTG特征已缓存: {cache_path}")
    print(f"CTG临床特征维度: {ctg_features.shape[1]}")

    fhrbs_norm = per_sample_normalize(fhrbs)
    fm_norm = per_sample_normalize(fm)
    uc_norm = per_sample_normalize(uc)

    all_stats = np.concatenate([fhr_stats, fm_stats, uc_stats, gest, age, ctg_features], axis=1)

    n = len(labels)
    indices = np.arange(n)
    idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=labels)
    labels_train_val = labels[idx_train_val]
    relative_val_size = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=relative_val_size, random_state=random_state, stratify=labels_train_val)

    stats_scaler = StandardScaler()
    stats_train = stats_scaler.fit_transform(all_stats[idx_train])
    stats_val = stats_scaler.transform(all_stats[idx_val])
    stats_test = stats_scaler.transform(all_stats[idx_test])

    x_train = {
        'fhrbs': fhrbs_norm[idx_train].reshape(-1, 1125, 1).astype(np.float32),
        'fm': fm_norm[idx_train].reshape(-1, 1125, 1).astype(np.float32),
        'uc': uc_norm[idx_train].reshape(-1, 1125, 1).astype(np.float32),
        'stats': stats_train.astype(np.float32),
    }
    x_val = {
        'fhrbs': fhrbs_norm[idx_val].reshape(-1, 1125, 1).astype(np.float32),
        'fm': fm_norm[idx_val].reshape(-1, 1125, 1).astype(np.float32),
        'uc': uc_norm[idx_val].reshape(-1, 1125, 1).astype(np.float32),
        'stats': stats_val.astype(np.float32),
    }
    x_test = {
        'fhrbs': fhrbs_norm[idx_test].reshape(-1, 1125, 1).astype(np.float32),
        'fm': fm_norm[idx_test].reshape(-1, 1125, 1).astype(np.float32),
        'uc': uc_norm[idx_test].reshape(-1, 1125, 1).astype(np.float32),
        'stats': stats_test.astype(np.float32),
    }

    y_train, y_val, y_test = labels[idx_train], labels[idx_val], labels[idx_test]

    print(f"\n数据划分完成:")
    print(f"  训练集: {len(y_train)} 样本, 分布: {np.bincount(y_train, minlength=num_classes)}")
    print(f"  验证集: {len(y_val)} 样本, 分布: {np.bincount(y_val, minlength=num_classes)}")
    print(f"  测试集: {len(y_test)} 样本, 分布: {np.bincount(y_test, minlength=num_classes)}")
    print(f"  总特征维度: {stats_train.shape[1]}")

    return x_train, x_val, x_test, y_train, y_val, y_test