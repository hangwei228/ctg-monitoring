# -*- coding: utf-8 -*-
"""
数据预处理方法 - 三分类版本 + Q值筛选 + 去噪

预处理流程：
  阶段0: 数据加载 (原始.dat文件 + CSV标签匹配)
  阶段1: Q值筛选 (阈值0.6)
  阶段2: 泓至去噪 (FHR: 自适应软阈值化, UC: 滑动平均)
  阶段3: FHR异常值/缺失值处理
  阶段4: UC/FM同步处理
  阶段5: 基线提取 (SG滤波 + EMD + K-Means)
  阶段6: 标准化 S(t) = F(t) - B(t)
  阶段7: 数据增强 (滑动窗口)
  阶段8: 长度统一 (900点)
  阶段9: 分层数据划分 (70/15/15)
"""

import os
import csv
import math
import numpy as np
import pandas as pd
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
    OUTPUT_PATH = r'./data/output/three_class_result'

    # 标签映射 (三分类)
    LABEL_MAP = {
        '有反应': 0,
        '无反应': 1,
        '可疑': 2
    }

    # FHR有效范围
    FHR_VALID_MIN = 40     
    FHR_VALID_MAX = 220     

    # 缺失段处理阈值
    MISSING_THRESHOLD = 5   

    # 最小信号长度
    MIN_SIGNAL_LENGTH = 750 

    # SG滤波参数
    SG_WINDOW = 7
    SG_POLYORDER = 2

    # EMD参数
    EMD_MAX_IMFS_TO_REMOVE = 0.5  

    # K-Means参数
    KMEANS_N_CLUSTERS = 2  

    # 目标长度
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
    """
    提取UC信号缺失段
    UC缺失定义: 连续15个点以上变化≤1的段
    """
    uc_missing = []
    uc_missing_start = -1
    uc_missing_end = -1

    for i, value in enumerate(uc_signal):
        if i > 0 and abs(value - uc_signal[i-1]) <= 1:
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
    """
    提取FHR信号缺失段 (来自曹珍Quality.py)
    FHR缺失定义: fhr < 50 或 fhr > 200 的连续段
    """
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
    """
    计算信号质量Q值 
    Q = 1 - (Σ(Xn·log₁₀(Xn)) + Σ(Ym·log₁₀(Ym))) / (2·l·log₁₀(l))
    """
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
    """软阈值化函数"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def calculate_adaptive_threshold(signal, percentile=75):
    """计算自适应阈值"""
    gradient = np.abs(np.diff(signal))
    if len(gradient) == 0:
        return 0
    threshold = np.percentile(gradient, percentile)
    return threshold * 0.1


def moving_average_denoise(signal, window=5):
    """滑动平均去噪"""
    if window <= 1 or len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def denoise_signal(fhr_signal, uc_signal):
    """对FHR和UC信号去噪 (泓至方法)"""
    # FHR去噪: 自适应软阈值化
    fhr_signal = np.nan_to_num(fhr_signal, nan=140)
    fhr_signal = np.clip(fhr_signal, 50, 220)
    threshold = calculate_adaptive_threshold(fhr_signal)
    fhr_denoised = soft_thresholding(fhr_signal, threshold)

    # UC去噪: 滑动平均
    uc_signal = np.nan_to_num(uc_signal, nan=0)
    uc_denoised = moving_average_denoise(uc_signal, window=3)

    return fhr_denoised, uc_denoised


# ==============================================================================
# 阶段0: 数据加载
# ==============================================================================
def read_dat_file(filepath):
    """
    读取.dat信号文件 (MyCTG_T格式)
    每条记录4字节: uint8 FHR + uint8 UC + uint16 FM
    采样率1Hz, 1500条记录 = 25分钟
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()

        n_records = len(data) // 4
        if n_records < 10:
            return None, None, None

        fhr = np.zeros(n_records, dtype=float)
        uc = np.zeros(n_records, dtype=float)
        fm = np.zeros(n_records, dtype=float)

        for i in range(n_records):
            offset = i * 4
            fhr[i] = data[offset]          # FHR: uint8, 0=缺失, 30-240 bpm
            uc[i] = data[offset + 1]       # UC: uint8, 0-100
            fm_raw = data[offset + 2] | (data[offset + 3] << 8)  # FM: uint16 LE
            fm[i] = 1.0 if (fm_raw & 0x90) != 0 else 0.0

        return fhr, uc, fm
    except Exception:
        return None, None, None


def parse_csv_labels(csv_path):
    """解析CSV文件获取标签和临床信息"""
    records = []
    encodings = ['gbk', 'gb2312', 'utf-8']

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                lines = f.readlines()

            if len(lines) < 3:
                continue

            # 跳过描述行，读取表头
            header_line = lines[1].strip()
            headers = [h.strip() for h in header_line.split(',')]

            # 找到关键列的索引
            id_idx = headers.index('编号') if '编号' in headers else None
            label_idx = headers.index('评价') if '评价' in headers else None

            # 可选列
            gest_age_idx = headers.index('孕周') if '孕周' in headers else None
            age_idx = headers.index('年龄') if '年龄' in headers else None
            duration_idx = headers.index('监护时长') if '监护时长' in headers else None

            for line in lines[2:]:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) <= max(filter(None, [id_idx, label_idx])):
                    continue

                record = {}
                if id_idx is not None and id_idx < len(parts):
                    record['编号'] = parts[id_idx].strip()
                if label_idx is not None and label_idx < len(parts):
                    record['评价'] = parts[label_idx].strip()
                if gest_age_idx is not None and gest_age_idx < len(parts):
                    record['孕周'] = parts[gest_age_idx].strip()
                if age_idx is not None and age_idx < len(parts):
                    record['年龄'] = parts[age_idx].strip()
                if duration_idx is not None and duration_idx < len(parts):
                    record['监护时长'] = parts[duration_idx].strip()

                if '编号' in record and '评价' in record:
                    records.append(record)

            if records:
                break
        except Exception:
            continue

    return records


def collect_data_pairs(config):
    """遍历原始数据集，收集.dat文件和对应的CSV标签"""
    print('='*70)
    print('[阶段0] 数据加载 - 收集.dat文件和CSV标签')
    print('='*70)

    data_pairs = []
    total_dat = 0
    matched = 0
    skipped_no_label = 0
    skipped_bad_signal = 0

    stages = ['第一阶段', '第二阶段', '第三阶段（主动学习）']

    for stage in stages:
        stage_path = os.path.join(config.RAW_DATA_PATH, stage)
        if not os.path.exists(stage_path):
            print(f'  跳过不存在的阶段: {stage}')
            continue

        print(f'\n  处理阶段: {stage}')

        # 遍历所有目录寻找Dat文件夹
        for root, dirs, files in os.walk(stage_path):
            if 'Dat' not in dirs:
                continue

            dat_dir = os.path.join(root, 'Dat')
            dat_files = sorted([f for f in os.listdir(dat_dir) if f.endswith('.dat')])

            if not dat_files:
                continue

            total_dat += len(dat_files)

            # 查找对应CSV文件 (当前目录或上级目录)
            csv_records = []
            csv_path = None

            # 当前目录
            for f in files:
                if f.endswith('.csv'):
                    csv_path = os.path.join(root, f)
                    csv_records = parse_csv_labels(csv_path)
                    break

            # 上级目录
            if not csv_records:
                parent_dir = os.path.dirname(root)
                if parent_dir != stage_path:
                    for f in os.listdir(parent_dir):
                        if f.endswith('.csv'):
                            csv_path = os.path.join(parent_dir, f)
                            csv_records = parse_csv_labels(csv_path)
                            break

            # 更上级目录
            if not csv_records:
                grandparent = os.path.dirname(os.path.dirname(root))
                if os.path.exists(grandparent):
                    for f in os.listdir(grandparent):
                        if f.endswith('.csv'):
                            csv_path = os.path.join(grandparent, f)
                            csv_records = parse_csv_labels(csv_path)
                            break

            if not csv_records:
                skipped_no_label += len(dat_files)
                continue

            # 建立编号->记录映射
            id_to_record = {}
            for rec in csv_records:
                rec_id = rec['编号'].strip()
                id_to_record[rec_id] = rec

            # 匹配.dat文件和CSV记录
            # 映射规则: dat编号 = CSV编号 × 10 + 1
            for dat_file in dat_files:
                dat_name = dat_file[:-4]  # 去掉.dat后缀
                try:
                    dat_num = int(dat_name)
                except ValueError:
                    skipped_bad_signal += 1
                    continue

                # 计算对应CSV编号
                csv_num = (dat_num - 1) // 10
                csv_id = str(csv_num)

                if csv_id not in id_to_record:
                    skipped_no_label += 1
                    continue

                record = id_to_record[csv_id]
                label_str = record['评价'].strip()

                if label_str not in config.LABEL_MAP:
                    skipped_no_label += 1
                    continue

                dat_path = os.path.join(dat_dir, dat_file)

                # 解析孕周和年龄
                gest_age = 0.0
                if '孕周' in record:
                    try:
                        ga_str = record['孕周'].strip()
                        if '+' in ga_str:
                            parts = ga_str.split('+')
                            gest_age = int(parts[0]) + int(parts[1]) / 4
                        elif ga_str:
                            gest_age = float(ga_str)
                    except:
                        gest_age = 0.0

                age = 0.0
                if '年龄' in record:
                    try:
                        age = float(record['年龄'].strip())
                    except:
                        age = 0.0

                data_pairs.append({
                    'dat_path': dat_path,
                    'label': config.LABEL_MAP[label_str],
                    'label_str': label_str,
                    'csv_id': csv_id,
                    'gest_age': gest_age,
                    'age': age
                })
                matched += 1

    print(f'\n  扫描.dat文件总数: {total_dat}')
    print(f'  成功匹配: {matched}')
    print(f'  未匹配(无标签): {skipped_no_label}')
    print(f'  未匹配(格式错误): {skipped_bad_signal}')

    # 统计标签分布
    label_counts = {}
    for pair in data_pairs:
        ls = pair['label_str']
        label_counts[ls] = label_counts.get(ls, 0) + 1
    print(f'  标签分布:')
    for name, cnt in sorted(label_counts.items()):
        print(f'    {name}: {cnt} ({cnt/len(data_pairs)*100:.1f}%)')

    return data_pairs


# ==============================================================================
# 阶段1: FHR异常值/缺失值处理 
# ==============================================================================
def stage1_fhr_cleaning(fhr_signal, config):
    """
    论文方法:
    1. f_n < 40bpm 或 (f_{n-1}==0 且 f_{n+1}==0) → 标记为NaN
    2. 删除首尾NaN
    3. 缺失段 > 5点: 删除并连接端点
    4. 缺失段 <= 5点: 中位数插值
    """
    fhr = fhr_signal.copy()

    if len(fhr) < 3:
        return None

    # Step 1: 标记异常值
    mask = np.zeros(len(fhr), dtype=bool)

    # f_n < 40bpm
    mask |= (fhr < config.FHR_VALID_MIN)

    # f_n > 合理上限
    mask |= (fhr > config.FHR_VALID_MAX)

    # 被0包围的点
    for i in range(1, len(fhr) - 1):
        if fhr[i-1] == 0 and fhr[i+1] == 0:
            mask[i] = True

    # 0值也标记为缺失
    mask |= (fhr == 0)

    fhr[mask] = np.nan

    # Step 2: 删除首尾NaN
    first_valid = 0
    while first_valid < len(fhr) and np.isnan(fhr[first_valid]):
        first_valid += 1

    last_valid = len(fhr) - 1
    while last_valid >= 0 and np.isnan(fhr[last_valid]):
        last_valid -= 1

    if first_valid > last_valid:
        return None

    fhr = fhr[first_valid:last_valid+1]

    # Step 3: 缺失段处理
    isnan = np.isnan(fhr)
    if not np.any(isnan):
        return fhr

    # 找到连续NaN段
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

    # 处理每个缺失段
    for start, end in segments_to_process:
        seg_len = end - start + 1
        valid_before = fhr[start-1] if start > 0 else None
        valid_after = fhr[end+1] if end < len(fhr) - 1 else None

        if seg_len > config.MISSING_THRESHOLD:
            # 删除缺失段，连接左右端点
            if valid_before is not None and valid_after is not None:
                # 线性插值连接
                fill_values = np.linspace(valid_before, valid_after, seg_len + 2)[1:-1]
                fhr[start:end+1] = fill_values
            elif valid_before is not None:
                fhr[start:end+1] = valid_before
            elif valid_after is not None:
                fhr[start:end+1] = valid_after
            else:
                fhr[start:end+1] = 0
        else:
            # 中位数插值
            valid_vals = fhr[~np.isnan(fhr)]
            if len(valid_vals) > 0:
                median_val = np.median(valid_vals)
                fhr[start:end+1] = median_val
            else:
                fhr[start:end+1] = 0

    # 处理剩余NaN
    fhr = np.nan_to_num(fhr, nan=0)

    # 长度过滤
    if len(fhr) < config.MIN_SIGNAL_LENGTH:
        return None

    return fhr


# ==============================================================================
# 阶段2: UC/FM同步处理
# ==============================================================================
def stage2_uc_fm_processing(fhr_clean, uc_raw, config):
    """
    UC: 与FHR同步，缺失值用中位数插值
    FM: 二值信号，缺失值插值为0
    """
    if uc_raw is None:
        uc_clean = np.zeros_like(fhr_clean)
    else:
        uc_clean = uc_raw.copy()
        # 长度对齐
        if len(uc_clean) < len(fhr_clean):
            uc_clean = np.pad(uc_clean, (0, len(fhr_clean) - len(uc_clean)), mode='edge')
        elif len(uc_clean) > len(fhr_clean):
            uc_clean = uc_clean[:len(fhr_clean)]

        # 中位数插值NaN
        nan_mask = np.isnan(uc_clean)
        if np.any(nan_mask):
            valid_vals = uc_clean[~nan_mask]
            if len(valid_vals) > 0:
                uc_clean[nan_mask] = np.median(valid_vals)
            else:
                uc_clean[nan_mask] = 0

        # 处理0值 (可能也是缺失)
        # 保留原始UC中的0值（宫缩信号可以是0）

    # FM = FHR - UC 
    fm_clean = fhr_clean - uc_clean

    return uc_clean, fm_clean


# ==============================================================================
# 阶段3: 基线提取
# ==============================================================================
def stage3_baseline_extraction(fhr_clean, config):
    """
    4步法:
    1. SG滤波降噪
    2. EMD分解 → 去除最高频IMF
    3. K-Means聚类极值点 → 基线拟合
    4. 回退机制: 如果基线提取失败，使用滑动中位数

    返回: (标准化信号, 基线)
    """
    if len(fhr_clean) < config.SG_WINDOW:
        return fhr_clean, np.full_like(fhr_clean, np.median(fhr_clean))

    # Step 1: SG滤波降噪
    try:
        fhr_sg = savgol_filter(fhr_clean, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        fhr_sg = fhr_clean.copy()

    # Step 2: EMD分解
    fhr_emd = fhr_sg.copy()
    try:
        emd = PyEMD()
        imfs = emd(fhr_sg)
        if imfs.shape[0] > 1:
            # 去除最高频IMF (第1个)，保留其余低频
            n_remove = max(1, int(imfs.shape[0] * 0.3))
            n_remove = min(n_remove, int(imfs.shape[0] * config.EMD_MAX_IMFS_TO_REMOVE))
            n_remove = max(1, n_remove)
            fhr_emd = np.sum(imfs[n_remove:], axis=0)
    except Exception:
        pass

    # Step 3: K-Means聚类极值点提取基线
    baseline = None
    try:
        # 找到所有极值点
        peaks = []
        troughs = []
        for i in range(1, len(fhr_emd) - 1):
            if fhr_emd[i] > fhr_emd[i-1] and fhr_emd[i] > fhr_emd[i+1]:
                peaks.append(i)
            elif fhr_emd[i] < fhr_emd[i-1] and fhr_emd[i] < fhr_emd[i+1]:
                troughs.append(i)

        extrema_indices = peaks + troughs

        if len(extrema_indices) >= config.KMEANS_N_CLUSTERS:
            extrema_values = fhr_emd[extrema_indices].reshape(-1, 1)

            kmeans = KMeans(n_clusters=config.KMEANS_N_CLUSTERS, random_state=0, n_init=10)
            kmeans.fit(extrema_values)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.flatten()

            # 选择值较小的聚类作为基线点
            # 基线是信号的低频慢变分量，通常在信号值的较低部分
            baseline_cluster = np.argmin(centers)
            baseline_indices = [extrema_indices[i] for i in range(len(labels)) if labels[i] == baseline_cluster]

            if len(baseline_indices) >= 2:
                baseline_indices = sorted(baseline_indices)
                baseline_values = fhr_emd[baseline_indices]

                # 线性插值得到完整基线
                baseline = np.interp(
                    np.arange(len(fhr_emd)),
                    baseline_indices,
                    baseline_values
                )
    except Exception:
        pass

    # Step 4: 回退机制 - 如果K-Means失败或结果不合理，使用滑动中位数
    if baseline is None:
        # 滑动中位数作为基线 (窗口=信号长度的10%)
        window = max(31, len(fhr_emd) // 10)
        if window % 2 == 0:
            window += 1
        try:
            baseline = median_filter(fhr_emd, size=window)
        except Exception:
            baseline = np.full_like(fhr_emd, np.median(fhr_emd))

    # 平滑基线
    try:
        baseline = savgol_filter(baseline, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        pass

    # 验证: 如果标准化后方差过小，改用简单中位数基线
    fhr_std = fhr_emd - baseline
    if np.std(fhr_std) < 0.1:
        baseline = np.full_like(fhr_emd, np.median(fhr_emd))
        fhr_std = fhr_emd - baseline

    return fhr_std, baseline


# ==============================================================================
# 阶段4: 标准化
# ==============================================================================
def stage4_standardization(fhr_clean, baseline):
    """S(t) = F(t) - B(t)"""
    return fhr_clean - baseline


# ==============================================================================
# 阶段5: 数据增强 
# ==============================================================================
def stage5_data_augmentation(fhr_std, uc_clean, fm_clean, label, config):
    """
    论文数据增强策略 (仅对非正常类):
    - 15-18min: 尾部截取15min (1部分)
    - 18-20min: 滑动3min截取15min (2部分)
    - =20min: 头部截取15min (1部分)
    - <15min: 丢弃
    """
    length = len(fhr_std)
    target = config.TARGET_LENGTH  # 900点 = 15min

    results = []

    if length < target:
        # 信号太短，丢弃
        return results

    if label == 0:
        # 正常类(有反应): 不做增强，取中心段
        start = max(0, length // 2 - target // 2)
        end = min(length, start + target)
        results.append({
            'fhr': fhr_std[start:end],
            'uc': uc_clean[start:end],
            'fm': fm_clean[start:end],
            'label': label
        })
    else:
        # 非正常类: 数据增强
        if length == target:
            # 恰好15min: 直接使用
            results.append({
                'fhr': fhr_std[:target],
                'uc': uc_clean[:target],
                'fm': fm_clean[:target],
                'label': label
            })
        elif length < target + config.AUGMENT_SLIDE:
            # 15-18min: 尾部截取
            results.append({
                'fhr': fhr_std[length-target:],
                'uc': uc_clean[length-target:],
                'fm': fm_clean[length-target:],
                'label': label
            })
        else:
            # 18min+: 滑动窗口截取多段
            n_segments = 1 + (length - target) // config.AUGMENT_SLIDE
            n_segments = min(n_segments, 3)  # 最多3段

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
    """统一信号长度到target_length"""
    L = len(fhr)

    if L >= target_length:
        start = L // 2 - target_length // 2
        fhr = fhr[start:start+target_length]
        uc = uc[start:start+target_length]
        fm = fm[start:start+target_length]
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
    """分层划分: 70% train, 30% test → train再分 80/20"""
    print('\n' + '='*70)
    print('[阶段7] 分层数据划分')
    print('='*70)

    indices = np.arange(len(labels_all))

    # 第一次划分: 70% train+val, 30% test
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels_all
    )

    # 第二次划分: train中取20%作为val
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=config.VAL_RATIO,
        random_state=config.RANDOM_STATE + 1,
        stratify=labels_all[idx_train_val]
    )

    print(f'  训练集: {len(idx_train)} 样本')
    print(f'  验证集: {len(idx_val)} 样本')
    print(f'  测试集: {len(idx_test)} 样本')

    # 标签分布
    for name, idx in [('训练集', idx_train), ('验证集', idx_val), ('测试集', idx_test)]:
        lbls = labels_all[idx]
        print(f'    {name}:')
        for lid, lname in [(0, '有反应'), (1, '无反应'), (2, '可疑')]:
            cnt = np.sum(lbls == lid)
            pct = cnt / len(lbls) * 100 if len(lbls) > 0 else 0
            print(f'      {lname}: {cnt} ({pct:.1f}%)')

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
    """保存处理后的数据"""
    print('\n' + '='*70)
    print('保存数据')
    print('='*70)
    print(f'  输出路径: {config.OUTPUT_PATH}')

    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    for split_name, data in split_data.items():
        print(f'\n  保存 {split_name}...')

        pd.DataFrame(data['fhr']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'fhr_{split_name}.csv'),
            header=None, index=False
        )
        pd.DataFrame(data['uc']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'uc_{split_name}.csv'),
            header=None, index=False
        )
        pd.DataFrame(data['fm']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'fm_{split_name}.csv'),
            header=None, index=False
        )
        pd.DataFrame(data['label']).to_csv(
            os.path.join(config.OUTPUT_PATH, f'label_{split_name}.csv'),
            header=None, index=False
        )

    print(f'\n  保存完成!')


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    config = Config()
    start_time = time.time()

    print('='*70)
    print('陈帆论文数据预处理方法 - 三分类版本')
    print('基于《基于BiGRU的自动判读产前胎心率宫缩信号模型研究》')
    print('='*70)
    print(f'原始数据路径: {config.RAW_DATA_PATH}')
    print(f'输出路径: {config.OUTPUT_PATH}')

    # ================================================================
    # 阶段0: 数据加载
    # ================================================================
    data_pairs = collect_data_pairs(config)

    if not data_pairs:
        print('错误: 没有找到有效的数据对!')
        return

    # ================================================================
    # 阶段1-8: Q值筛选 + 去噪 + 信号预处理
    # ================================================================
    print('\n' + '='*70)
    print('Q值筛选 + 泓至去噪 + 信号预处理')
    print('='*70)

    all_fhr = []
    all_uc = []
    all_fm = []
    all_labels = []

    processed = 0
    failed = 0
    augmented_total = 0
    skipped_low_q = 0

    for i, pair in enumerate(data_pairs):
        if i % 500 == 0:
            print(f'  处理进度: {i}/{len(data_pairs)} (已成功: {processed}, 失败: {failed}, Q值剔除: {skipped_low_q})')

        # 读取.dat信号文件
        fhr_raw, uc_raw, fm_raw = read_dat_file(pair['dat_path'])

        if fhr_raw is None or len(fhr_raw) < 10:
            failed += 1
            continue

        # Q值筛选
        q_value = get_Q(fhr_raw, uc_raw)
        if q_value < config.Q_THRESHOLD:
            skipped_low_q += 1
            continue

        # 去噪
        fhr_denoised, uc_denoised = denoise_signal(fhr_raw, uc_raw)

        # 阶段1: FHR异常值/缺失值处理
        fhr_clean = stage1_fhr_cleaning(fhr_denoised, config)
        if fhr_clean is None:
            failed += 1
            continue

        # 阶段2: UC/FM处理
        if uc_denoised is not None and len(uc_denoised) == len(fhr_clean):
            uc_clean = uc_denoised
        else:
            uc_clean = np.zeros_like(fhr_clean)
        if fm_raw is not None and len(fm_raw) == len(fhr_clean):
            fm_clean = fm_raw
        else:
            fm_clean = np.zeros_like(fhr_clean)

        # 阶段3: 基线提取 + 阶段4: 标准化
        fhr_std, baseline = stage3_baseline_extraction(fhr_clean, config)

        # 阶段5: 数据增强
        augmented = stage5_data_augmentation(
            fhr_std, uc_clean, fm_clean, pair['label'], config
        )

        if not augmented:
            failed += 1
            continue

        # 阶段6: 长度统一
        for aug in augmented:
            fhr_final, uc_final, fm_final = stage6_length_normalization(
                aug['fhr'], aug['uc'], aug['fm'], config.TARGET_LENGTH
            )
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

    # 标签分布
    print(f'    标签分布:')
    for lid, lname in [(0, '有反应'), (1, '无反应'), (2, '可疑')]:
        cnt = sum(1 for l in all_labels if l == lid)
        print(f'      {lname}: {cnt} ({cnt/len(all_labels)*100:.1f}%)')

    # 转换为数组
    all_fhr = np.array(all_fhr)
    all_uc = np.array(all_uc)
    all_fm = np.array(all_fm)
    all_labels = np.array(all_labels)

    # ================================================================
    # 阶段7: 数据划分
    # ================================================================
    split_data = stage7_stratified_split(all_fhr, all_uc, all_fm, all_labels, config)

    # ================================================================
    # 保存
    # ================================================================
    save_data(split_data, config)

    elapsed = time.time() - start_time
    print('\n' + '='*70)
    print(f'预处理完成! 总耗时: {elapsed/60:.1f} 分钟')
    print('='*70)


if __name__ == '__main__':
    main()
