# -*- coding: utf-8 -*-
"""
CTU-UHB 数据集预处理 - 二分类版本 (基于 pH 阈值)
方法: Q值筛选 + 去噪 + FHR异常值处理 + 基线提取 + 标准化 + 数据增强 + 长度统一

数据来源: CTU-UHB Intrapartum CTG Database (physionet: ctu-uhb-ctgdb)
  - 每条记录 .dat + .hea (WFDB格式), FHR 与 UC 采样率 4Hz
  - .hea 注释中含 #pH (脐带血 pH)
  - 本脚本先下采样到 1Hz, 再复用原 1Hz 预处理流程
  - 二分类标签: pH >= 7.15 -> 0(正常); pH < 7.15 -> 1(酸血症)
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from PyEMD import EMD as PyEMD
import warnings
import time

warnings.filterwarnings('ignore')


# ==============================================================================
# 配置参数
# ==============================================================================
class Config:
    # 原始数据路径
    RAW_DATA_PATH = r'./data/raw'

    # 输出路径
    OUTPUT_PATH =  r'./data/output/binary_result'

    # 分类模式
    CLASS_NAMES = {0: '正常', 1: '酸血症'}
    NUM_CLASSES = 2

    # pH 阈值 (二分类)
    PH_ACIDEMIA = 7.15

    # 原始采样率 -> 目标采样率
    SRC_FS = 4
    DST_FS = 1
    DOWNSAMPLE_FACTOR = SRC_FS // DST_FS  # 4

    # FHR有效范围
    FHR_VALID_MIN = 40
    FHR_VALID_MAX = 220

    # 缺失段处理阈值
    MISSING_THRESHOLD = 5

    # 最小信号长度 (1Hz)
    MIN_SIGNAL_LENGTH = 750

    # SG滤波参数
    SG_WINDOW = 7
    SG_POLYORDER = 2

    # EMD参数
    EMD_MAX_IMFS_TO_REMOVE = 0.5

    # K-Means参数
    KMEANS_N_CLUSTERS = 2

    # 目标长度 (1Hz, 15分钟=900点)
    TARGET_LENGTH = 900

    # 数据增强滑动步长
    AUGMENT_SLIDE = 180

    # 数据划分
    TEST_SIZE = 0.3
    VAL_RATIO = 0.2
    RANDOM_STATE = 42

    # Q值筛选 
    Q_THRESHOLD = 0.6


# ==============================================================================
# Q值计算
# ==============================================================================
def get_Ym(uc_signal):
    """提取UC信号缺失段: 连续15个点以上变化<=1的段"""
    uc_missing = []
    uc_missing_start = -1
    uc_missing_end = -1

    for i, value in enumerate(uc_signal):
        if i > 0 and abs(value - uc_signal[i - 1]) <= 1:
            if uc_missing_start == -1:
                uc_missing_start = i - 1
            uc_missing_end = i
        else:
            if uc_missing_start != -1:
                uc_missing.append((uc_missing_start, uc_missing_end))
                uc_missing_start = -1
                uc_missing_end = -1

    if uc_missing_start != -1:
        uc_missing.append((uc_missing_start, uc_missing_end))

    uc_missing_lengths = [end - start + 1 for start, end in uc_missing]
    Ym = [l for l in uc_missing_lengths if l > 15]
    return Ym


def get_Xm(fhr_signal):
    """提取FHR信号缺失段: fhr<50 或 fhr>200 的连续段"""
    fhr_missing = []
    fhr_missing_start = -1
    fhr_missing_end = -1

    for i, value in enumerate(fhr_signal):
        if (value < 50 or value > 200) and i > 0:
            if fhr_missing_start == -1:
                fhr_missing_start = i - 1
            fhr_missing_end = i
        else:
            if fhr_missing_start != -1:
                fhr_missing.append((fhr_missing_start, fhr_missing_end))
                fhr_missing_start = -1
                fhr_missing_end = -1

    if fhr_missing_start != -1:
        fhr_missing.append((fhr_missing_start, fhr_missing_end))

    fhr_missing_lengths = [end - start + 1 for start, end in fhr_missing]
    Xn = [l for l in fhr_missing_lengths if l > 0]
    return Xn


def get_Q(fhr_signal, uc_signal):
    """计算信号质量Q值"""
    l = len(uc_signal)
    if l <= 1:
        return 0

    Y = get_Ym(uc_signal)
    X = get_Xm(fhr_signal)

    Xn_log_Xn_sum = sum([(Xn * math.log10(Xn)) for Xn in X]) if X else 0
    Ym_log_Ym_sum = sum([(Ym * math.log10(Ym)) for Ym in Y]) if Y else 0

    if l <= 0 or math.log10(l) == 0:
        return 0

    Q = 1 - ((Xn_log_Xn_sum + Ym_log_Ym_sum) / (2 * l * math.log10(l)))
    return max(0, min(1, Q))


# ==============================================================================
# 去噪方法
# ==============================================================================
def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def calculate_adaptive_threshold(signal, percentile=75):
    gradient = np.abs(np.diff(signal))
    if len(gradient) == 0:
        return 0
    threshold = np.percentile(gradient, percentile)
    return threshold * 0.1


def moving_average_denoise(signal, window=5):
    if window <= 1 or len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def denoise_signal(fhr_signal, uc_signal):
    fhr_signal = np.nan_to_num(fhr_signal, nan=140)
    fhr_signal = np.clip(fhr_signal, 50, 220)
    threshold = calculate_adaptive_threshold(fhr_signal)
    fhr_denoised = soft_thresholding(fhr_signal, threshold)

    uc_signal = np.nan_to_num(uc_signal, nan=0)
    uc_denoised = moving_average_denoise(uc_signal, window=3)

    return fhr_denoised, uc_denoised


# ==============================================================================
# 数据加载 (CTU-UHB WFDB格式)
# ==============================================================================
def parse_ph(hea_path):
    """从 .hea 注释中解析 #pH 值"""
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s.startswith('#') and 'pH' in s:
                    # 形如: #pH           7.14
                    parts = s[1:].split()
                    # parts[0]=='pH', parts[1]==数值
                    for tok in parts[1:]:
                        try:
                            return float(tok)
                        except ValueError:
                            continue
    except Exception:
        return None
    return None


def read_ctg_record(dat_path, config):
    """
    读取 ctu-uhb 记录 (WFDB .dat).
    返回 (fhr_4hz, uc_4hz, ph) 或 (None, None, None)
    """
    rec_path = dat_path[:-4] if dat_path.endswith('.dat') else dat_path
    try:
        rec = wfdb.rdrecord(rec_path)
    except Exception:
        return None, None, None

    names = [n.strip() for n in rec.sig_name]
    if 'FHR' not in names or 'UC' not in names:
        return None, None, None

    fhr = rec.p_signal[:, names.index('FHR')].astype(float).copy()
    uc = rec.p_signal[:, names.index('UC')].astype(float).copy()

    # FHR 缺失标记: <=0 或 >220 -> NaN
    fhr = np.where((fhr <= 0) | (fhr > config.FHR_VALID_MAX), np.nan, fhr)
    # UC: 0 为合法 (无宫缩); 仅把 NaN 置 0
    uc = np.nan_to_num(uc, nan=0.0)

    hea_path = dat_path
    if hea_path.endswith('.dat'):
        hea_path = hea_path[:-4] + '.hea'
    else:
        hea_path = hea_path + '.hea'
    ph = parse_ph(hea_path)
    if ph is None:
        return None, None, None

    return fhr, uc, ph


def downsample(sig, factor):
    """非重叠窗口平均下采样 (窗口内 NaN 忽略)"""
    n = len(sig)
    n_win = n // factor
    if n_win == 0:
        return np.array([])
    s = sig[:n_win * factor].reshape(n_win, factor)
    return np.nanmean(s, axis=1)


def ph_to_label(ph, config):
    """二分类: pH >= 阈值 -> 0(正常); 否则 -> 1(酸血症)"""
    return 0 if ph >= config.PH_ACIDEMIA else 1


def collect_ctg_data(config):
    print('=' * 70)
    print('[阶段0] 数据加载 - CTU-UHB (WFDB)')
    print('=' * 70)

    dat_files = sorted(glob.glob(os.path.join(config.RAW_DATA_PATH, '*.dat')))
    print(f'  发现 .dat 文件: {len(dat_files)}')

    data_pairs = []
    skipped_noph = 0
    skipped_short = 0

    for dp in dat_files:
        fhr_4hz, uc_4hz, ph = read_ctg_record(dp, config)
        if fhr_4hz is None:
            continue

        fhr_1hz = downsample(fhr_4hz, config.DOWNSAMPLE_FACTOR)
        uc_1hz = downsample(uc_4hz, config.DOWNSAMPLE_FACTOR)

        if len(fhr_1hz) < config.MIN_SIGNAL_LENGTH:
            skipped_short += 1
            continue

        label = ph_to_label(ph, config)
        data_pairs.append({
            'dat_path': dp,
            'fhr_1hz': fhr_1hz,
            'uc_1hz': uc_1hz,
            'ph': ph,
            'label': label,
        })

    print(f'  有效样本(含pH且长度足够): {len(data_pairs)}')
    print(f'  跳过(无pH): {skipped_noph}  跳过(过短): {skipped_short}')

    # 标签分布
    label_counts = {}
    for p in data_pairs:
        label_counts[p['label']] = label_counts.get(p['label'], 0) + 1
    print('  标签分布:')
    for lid in sorted(label_counts):
        name = config.CLASS_NAMES.get(lid, str(lid))
        cnt = label_counts[lid]
        print(f'    {name}({lid}): {cnt} ({cnt / len(data_pairs) * 100:.1f}%)')

    return data_pairs


# ==============================================================================
# 阶段1+2: FHR/UC 联合清洗 
# ==============================================================================
def _fill_uc(uc):
    nan_mask = np.isnan(uc)
    if np.any(nan_mask):
        valid = uc[~nan_mask]
        uc[nan_mask] = np.median(valid) if len(valid) > 0 else 0
    return uc


def stage1_fhr_uc_cleaning(fhr_signal, uc_signal, config):
    fhr = fhr_signal.copy()
    uc = uc_signal.copy()

    if len(fhr) < 3:
        return None, None

    mask = np.zeros(len(fhr), dtype=bool)
    mask |= (fhr < config.FHR_VALID_MIN)
    mask |= (fhr > config.FHR_VALID_MAX)
    for i in range(1, len(fhr) - 1):
        if fhr[i - 1] == 0 and fhr[i + 1] == 0:
            mask[i] = True
    mask |= (fhr == 0)
    fhr[mask] = np.nan
    uc[mask] = np.nan

    # 删除首尾 NaN
    first_valid = 0
    while first_valid < len(fhr) and np.isnan(fhr[first_valid]):
        first_valid += 1
    last_valid = len(fhr) - 1
    while last_valid >= 0 and np.isnan(fhr[last_valid]):
        last_valid -= 1

    if first_valid > last_valid:
        return None, None

    fhr = fhr[first_valid:last_valid + 1]
    uc = uc[first_valid:last_valid + 1]

    # 缺失段处理 (内部)
    isnan = np.isnan(fhr)
    if np.any(isnan):
        nan_start = None
        segments_to_process = []
        for i in range(len(fhr)):
            if isnan[i]:
                if nan_start is None:
                    nan_start = i
            else:
                if nan_start is not None:
                    segments_to_process.append((nan_start, i - 1))
                    nan_start = None
        if nan_start is not None:
            segments_to_process.append((nan_start, len(fhr) - 1))

        for start, end in segments_to_process:
            seg_len = end - start + 1
            valid_before = fhr[start - 1] if start > 0 else None
            valid_after = fhr[end + 1] if end < len(fhr) - 1 else None

            if seg_len > config.MISSING_THRESHOLD:
                if valid_before is not None and valid_after is not None:
                    fill_values = np.linspace(valid_before, valid_after, seg_len + 2)[1:-1]
                    fhr[start:end + 1] = fill_values
                elif valid_before is not None:
                    fhr[start:end + 1] = valid_before
                elif valid_after is not None:
                    fhr[start:end + 1] = valid_after
                else:
                    fhr[start:end + 1] = 0
            else:
                valid_vals = fhr[~np.isnan(fhr)]
                fhr[start:end + 1] = np.median(valid_vals) if len(valid_vals) > 0 else 0

    fhr = np.nan_to_num(fhr, nan=0)
    uc = _fill_uc(uc)

    if len(fhr) < config.MIN_SIGNAL_LENGTH:
        return None, None

    return fhr, uc


# ==============================================================================
# 阶段3: 基线提取
# ==============================================================================
def stage3_baseline_extraction(fhr_clean, config):
    if len(fhr_clean) < config.SG_WINDOW:
        return fhr_clean, np.full_like(fhr_clean, np.median(fhr_clean))

    try:
        fhr_sg = savgol_filter(fhr_clean, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        fhr_sg = fhr_clean.copy()

    fhr_emd = fhr_sg.copy()
    try:
        emd = PyEMD()
        imfs = emd(fhr_sg)
        if imfs.shape[0] > 1:
            n_remove = max(1, int(imfs.shape[0] * 0.3))
            n_remove = min(n_remove, int(imfs.shape[0] * config.EMD_MAX_IMFS_TO_REMOVE))
            n_remove = max(1, n_remove)
            fhr_emd = np.sum(imfs[n_remove:], axis=0)
    except Exception:
        pass

    baseline = None
    try:
        peaks = []
        troughs = []
        for i in range(1, len(fhr_emd) - 1):
            if fhr_emd[i] > fhr_emd[i - 1] and fhr_emd[i] > fhr_emd[i + 1]:
                peaks.append(i)
            elif fhr_emd[i] < fhr_emd[i - 1] and fhr_emd[i] < fhr_emd[i + 1]:
                troughs.append(i)

        extrema_indices = peaks + troughs

        if len(extrema_indices) >= config.KMEANS_N_CLUSTERS:
            extrema_values = fhr_emd[extrema_indices].reshape(-1, 1)
            kmeans = KMeans(n_clusters=config.KMEANS_N_CLUSTERS, random_state=0, n_init=10)
            kmeans.fit(extrema_values)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.flatten()
            baseline_cluster = np.argmin(centers)
            baseline_indices = [extrema_indices[i] for i in range(len(labels)) if labels[i] == baseline_cluster]

            if len(baseline_indices) >= 2:
                baseline_indices = sorted(baseline_indices)
                baseline_values = fhr_emd[baseline_indices]
                baseline = np.interp(np.arange(len(fhr_emd)), baseline_indices, baseline_values)
    except Exception:
        pass

    if baseline is None:
        window = max(31, len(fhr_emd) // 10)
        if window % 2 == 0:
            window += 1
        try:
            baseline = median_filter(fhr_emd, size=window)
        except Exception:
            baseline = np.full_like(fhr_emd, np.median(fhr_emd))

    try:
        baseline = savgol_filter(baseline, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        pass

    fhr_std = fhr_emd - baseline
    if np.std(fhr_std) < 0.1:
        baseline = np.full_like(fhr_emd, np.median(fhr_emd))
        fhr_std = fhr_emd - baseline

    return fhr_std, baseline


# ==============================================================================
# 阶段5: 数据增强
# ==============================================================================
def stage5_data_augmentation(fhr_std, uc_clean, fm_clean, label, config):
    length = len(fhr_std)
    target = config.TARGET_LENGTH

    results = []

    if length < target:
        return results

    if label == 0:
        # 正常类: 取中心段
        start = max(0, length // 2 - target // 2)
        end = min(length, start + target)
        results.append({
            'fhr': fhr_std[start:end],
            'uc': uc_clean[start:end],
            'fm': fm_clean[start:end],
            'label': label
        })
    else:
        if length == target:
            results.append({
                'fhr': fhr_std[:target],
                'uc': uc_clean[:target],
                'fm': fm_clean[:target],
                'label': label
            })
        elif length < target + config.AUGMENT_SLIDE:
            results.append({
                'fhr': fhr_std[length - target:],
                'uc': uc_clean[length - target:],
                'fm': fm_clean[length - target:],
                'label': label
            })
        else:
            n_segments = 1 + (length - target) // config.AUGMENT_SLIDE
            n_segments = min(n_segments, 3)
            for seg in range(n_segments):
                start = seg * config.AUGMENT_SLIDE
                end = start + target
                if end <= length:
                    results.append({
                        'fhr': fhr_std[start:end],
                        'uc': uc_clean[start:end],
                        'fm': fm_clean[start:end],
                        'label': label
                    })

    return results


# ==============================================================================
# 阶段6: 长度统一
# ==============================================================================
def stage6_length_normalization(fhr, uc, fm, target_length):
    L = len(fhr)
    if L >= target_length:
        start = L // 2 - target_length // 2
        fhr = fhr[start:start + target_length]
        uc = uc[start:start + target_length]
        fm = fm[start:start + target_length]
    else:
        pad = target_length - L
        fhr = np.pad(fhr, (0, pad), mode='edge')
        uc = np.pad(uc, (0, pad), mode='edge')
        fm = np.pad(fm, (0, pad), mode='edge')
    return fhr, uc, fm


# ==============================================================================
# 阶段7: 分层数据划分
# ==============================================================================
def stage7_stratified_split(fhr_all, uc_all, fm_all, labels_all, config):
    print('\n' + '=' * 70)
    print('[阶段7] 分层数据划分')
    print('=' * 70)

    indices = np.arange(len(labels_all))

    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels_all
    )

    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=config.VAL_RATIO,
        random_state=config.RANDOM_STATE + 1,
        stratify=labels_all[idx_train_val]
    )

    print(f'  训练集: {len(idx_train)} 样本')
    print(f'  验证集: {len(idx_val)} 样本')
    print(f'  测试集: {len(idx_test)} 样本')

    for lid in sorted(config.CLASS_NAMES.keys()):
        name = config.CLASS_NAMES[lid]
        for split_name, idx in [('训练集', idx_train), ('验证集', idx_val), ('测试集', idx_test)]:
            lbls = labels_all[idx]
            cnt = np.sum(lbls == lid)
            pct = cnt / len(lbls) * 100 if len(lbls) > 0 else 0
            print(f'    {name}({lid}) - {split_name}: {cnt} ({pct:.1f}%)')

    return {
        'train': {
            'fhr': fhr_all[idx_train],
            'uc': uc_all[idx_train],
            'fm': fm_all[idx_train],
            'label': labels_all[idx_train]
        },
        'val': {
            'fhr': fhr_all[idx_val],
            'uc': uc_all[idx_val],
            'fm': fm_all[idx_val],
            'label': labels_all[idx_val]
        },
        'test': {
            'fhr': fhr_all[idx_test],
            'uc': uc_all[idx_test],
            'fm': fm_all[idx_test],
            'label': labels_all[idx_test]
        }
    }


# ==============================================================================
# 保存数据
# ==============================================================================
def save_data(split_data, config):
    print('\n' + '=' * 70)
    print('保存数据')
    print('=' * 70)
    print(f'  输出路径: {config.OUTPUT_PATH}')

    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    for split_name, data in split_data.items():
        print(f'\n  保存 {split_name}...')
        pd.DataFrame(data['fhr']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'fhr_{split_name}.csv'), header=None, index=False)
        pd.DataFrame(data['uc']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'uc_{split_name}.csv'), header=None, index=False)
        pd.DataFrame(data['fm']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'fm_{split_name}.csv'), header=None, index=False)
        pd.DataFrame(data['label']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'label_{split_name}.csv'), header=None, index=False)

    print(f'\n  保存完成!')


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    config = Config()
    start_time = time.time()

    print('=' * 70)
    print('CTU-UHB 数据预处理 - 二分类 (pH阈值)')
    print('=' * 70)
    print(f'原始数据路径: {config.RAW_DATA_PATH}')
    print(f'输出路径: {config.OUTPUT_PATH}')
    print(f'pH阈值: <{config.PH_ACIDEMIA} 判为酸血症(1), 否则正常(0)')

    data_pairs = collect_ctg_data(config)
    if not data_pairs:
        print('错误: 没有找到有效的数据对!')
        return

    print('\n' + '=' * 70)
    print('Q值筛选 + 去噪 + 信号预处理')
    print('=' * 70)

    all_fhr = []
    all_uc = []
    all_fm = []
    all_labels = []

    processed = 0
    failed = 0
    augmented_total = 0
    skipped_low_q = 0

    for i, pair in enumerate(data_pairs):
        if i % 50 == 0:
            print(f'  处理进度: {i}/{len(data_pairs)} (成功:{processed}, 失败:{failed}, Q剔除:{skipped_low_q})')

        fhr_1hz = pair['fhr_1hz']
        uc_1hz = pair['uc_1hz']

        # Q值筛选 (下采样后的1Hz信号, NaN置0以计入缺失)
        fhr_q = np.nan_to_num(fhr_1hz, nan=0)
        uc_q = np.nan_to_num(uc_1hz, nan=0)
        q_value = get_Q(fhr_q, uc_q)
        if q_value < config.Q_THRESHOLD:
            skipped_low_q += 1
            continue

        # 去噪
        fhr_denoised, uc_denoised = denoise_signal(fhr_1hz, uc_1hz)

        # FHR + UC 联合清洗
        fhr_clean, uc_clean = stage1_fhr_uc_cleaning(fhr_denoised, uc_denoised, config)
        if fhr_clean is None:
            failed += 1
            continue

        # FM = FHR - UC
        fm_clean = fhr_clean - uc_clean

        # 基线提取 + 标准化
        fhr_std, baseline = stage3_baseline_extraction(fhr_clean, config)

        # 数据增强
        augmented = stage5_data_augmentation(fhr_std, uc_clean, fm_clean, pair['label'], config)
        if not augmented:
            failed += 1
            continue

        # 长度统一
        for aug in augmented:
            fhr_final, uc_final, fm_final = stage6_length_normalization(
                aug['fhr'], aug['uc'], aug['fm'], config.TARGET_LENGTH)
            all_fhr.append(fhr_final)
            all_uc.append(uc_final)
            all_fm.append(fm_final)
            all_labels.append(aug['label'])
            augmented_total += 1

        processed += 1

    print(f'\n  处理完成:')
    print(f'    原始样本: {len(data_pairs)}')
    print(f'    成功处理: {processed}')
    print(f'    失败丢弃: {failed}')
    print(f'    Q值筛选剔除(Q<{config.Q_THRESHOLD}): {skipped_low_q}')
    print(f'    增强后总数: {augmented_total}')

    print('    标签分布:')
    for lid in sorted(config.CLASS_NAMES.keys()):
        name = config.CLASS_NAMES[lid]
        cnt = sum(1 for l in all_labels if l == lid)
        print(f'      {name}({lid}): {cnt} ({cnt / max(1, len(all_labels)) * 100:.1f}%)')

    all_fhr = np.array(all_fhr)
    all_uc = np.array(all_uc)
    all_fm = np.array(all_fm)
    all_labels = np.array(all_labels)

    split_data = stage7_stratified_split(all_fhr, all_uc, all_fm, all_labels, config)
    save_data(split_data, config)

    elapsed = time.time() - start_time
    print('\n' + '=' * 70)
    print(f'预处理完成! 总耗时: {elapsed / 60:.1f} 分钟')
    print('=' * 70)


if __name__ == '__main__':
    main()
