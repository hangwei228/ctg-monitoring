# -*- coding: utf-8 -*-
"""
产前 CTG 信号预处理 - 二分类版本

与三分类的区别: 将 "无反应" 与 "可疑" 合并为异常类 (标签 1)。

流程: 数据加载 -> Q值筛选 -> 异常/缺失处理 -> UC/FM同步 ->
基线提取(EMD+KMeans) -> 标准化 S=F-B -> 数据增强 -> 长度统一 -> 分层划分

注: 信号去噪由分类模型中的 EVT-ADB 模块完成, 本脚本不做去噪。
"""

import os
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


class Config:
    RAW_DATA_PATH = r'./raw_data'                                  # 原始 .dat + 标签 CSV
    OUTPUT_PATH = r'./output/binary'                               # 输出目录
    LABEL_MAP = {'有反应': 0, '无反应': 1, '可疑': 1}              # 异常类合并
    FHR_VALID_MIN = 40
    FHR_VALID_MAX = 220
    MISSING_THRESHOLD = 5
    MIN_SIGNAL_LENGTH = 750
    SG_WINDOW = 7
    SG_POLYORDER = 2
    EMD_MAX_IMFS_TO_REMOVE = 0.5
    KMEANS_N_CLUSTERS = 2
    TARGET_LENGTH = 900
    AUGMENT_SLIDE = 180
    TEST_SIZE = 0.3
    VAL_RATIO = 0.2
    RANDOM_STATE = 42
    Q_THRESHOLD = 0.6


# ---------- 信号质量 Q 值 ----------
def get_Ym(uc_signal):
    uc_missing = []
    start = -1
    for i in range(1, len(uc_signal)):
        if abs(uc_signal[i] - uc_signal[i - 1]) <= 1:
            if start == -1:
                start = i - 1
        elif start != -1:
            uc_missing.append((start, i - 1))
            start = -1
    if start != -1:
        uc_missing.append((start, len(uc_signal) - 1))
    return [e - s + 1 for s, e in uc_missing if e - s + 1 > 15]


def get_Xm(fhr_signal):
    miss = []
    start = -1
    for i in range(1, len(fhr_signal)):
        if (fhr_signal[i] < 50 or fhr_signal[i] > 200):
            if start == -1:
                start = i - 1
        elif start != -1:
            miss.append((start, i - 1))
            start = -1
    if start != -1:
        miss.append((start, len(fhr_signal) - 1))
    return [e - s + 1 for s, e in miss if e - s + 1 > 0]


def get_Q(fhr_signal, uc_signal):
    l = len(uc_signal)
    if l <= 1 or math.log10(l) == 0:
        return 0
    Y = get_Ym(uc_signal)
    X = get_Xm(fhr_signal)
    Xs = sum(x * math.log10(x) for x in X) if X else 0
    Ys = sum(y * math.log10(y) for y in Y) if Y else 0
    Q = 1 - (Xs + Ys) / (2 * l * math.log10(l))
    return max(0, min(1, Q))


# ---------- 数据加载 ----------
def read_dat_file(filepath):
    """读取 MyCTG_T 格式 .dat: 每记录4字节 FHR(uint8)/UC(uint8)/FM(uint16)"""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        n = len(data) // 4
        if n < 10:
            return None, None, None
        fhr = np.zeros(n, dtype=float)
        uc = np.zeros(n, dtype=float)
        fm = np.zeros(n, dtype=float)
        for i in range(n):
            o = i * 4
            fhr[i] = data[o]
            uc[i] = data[o + 1]
            fm_raw = data[o + 2] | (data[o + 3] << 8)
            fm[i] = 1.0 if (fm_raw & 0x90) != 0 else 0.0
        return fhr, uc, fm
    except Exception:
        return None, None, None


def parse_csv_labels(csv_path):
    records = []
    for enc in ['gbk', 'gb2312', 'utf-8']:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            if len(lines) < 3:
                continue
            headers = [h.strip() for h in lines[1].strip().split(',')]
            idx = {k: headers.index(k) for k in ['编号', '评价', '孕周', '年龄', '监护时长'] if k in headers}
            for line in lines[2:]:
                parts = [p.strip() for p in line.strip().split(',')]
                if '编号' not in idx or '评价' not in idx:
                    continue
                if len(parts) <= max(idx['编号'], idx['评价']):
                    continue
                rec = {'编号': parts[idx['编号']], '评价': parts[idx['评价']]}
                for k in ('孕周', '年龄', '监护时长'):
                    if k in idx and idx[k] < len(parts):
                        rec[k] = parts[idx[k]]
                records.append(rec)
            if records:
                break
        except Exception:
            continue
    return records


def collect_data_pairs(config):
    print('[阶段0] 收集 .dat 与标签')
    pairs, total, matched, no_label, bad = [], 0, 0, 0, 0
    for stage in ['第一阶段', '第二阶段', '第三阶段（主动学习）']:
        sp = os.path.join(config.RAW_DATA_PATH, stage)
        if not os.path.exists(sp):
            continue
        for root, dirs, files in os.walk(sp):
            if 'Dat' not in dirs:
                continue
            dat_dir = os.path.join(root, 'Dat')
            dat_files = sorted(f for f in os.listdir(dat_dir) if f.endswith('.dat'))
            total += len(dat_files)
            recs = []
            for f in files:
                if f.endswith('.csv'):
                    recs = parse_csv_labels(os.path.join(root, f))
                    break
            if not recs:
                for d in (os.path.dirname(root), os.path.dirname(os.path.dirname(root))):
                    if os.path.exists(d):
                        for f in os.listdir(d):
                            if f.endswith('.csv'):
                                recs = parse_csv_labels(os.path.join(d, f))
                                break
                    if recs:
                        break
            if not recs:
                no_label += len(dat_files)
                continue
            id_map = {r['编号'].strip(): r for r in recs}
            for df in dat_files:
                name = df[:-4]
                try:
                    num = int(name)
                except ValueError:
                    bad += 1
                    continue
                cid = str((num - 1) // 10)
                if cid not in id_map:
                    no_label += 1
                    continue
                r = id_map[cid]
                lab = r['评价'].strip()
                if lab not in config.LABEL_MAP:
                    no_label += 1
                    continue
                ga = 0.0
                if '孕周' in r:
                    try:
                        g = r['孕周'].strip()
                        ga = float(g.split('+')[0]) + (float(g.split('+')[1]) / 4 if '+' in g else 0)
                    except Exception:
                        ga = 0.0
                age = 0.0
                if '年龄' in r:
                    try:
                        age = float(r['年龄'].strip())
                    except Exception:
                        age = 0.0
                pairs.append({'dat_path': os.path.join(dat_dir, df),
                              'label': config.LABEL_MAP[lab], 'label_str': lab,
                              'gest_age': ga, 'age': age})
                matched += 1
    print(f'  扫描 {total}, 匹配 {matched}, 无标签 {no_label}, 格式错误 {bad}')
    return pairs


# ---------- 信号清洗 ----------
def stage1_fhr_cleaning(fhr, config):
    if len(fhr) < 3:
        return None
    mask = (fhr < config.FHR_VALID_MIN) | (fhr > config.FHR_VALID_MAX) | (fhr == 0)
    for i in range(1, len(fhr) - 1):
        if fhr[i - 1] == 0 and fhr[i + 1] == 0:
            mask[i] = True
    fhr = fhr.copy()
    fhr[mask] = np.nan
    fv = np.where(~np.isnan(fhr))[0]
    if len(fv) == 0:
        return None
    fhr = fhr[fv[0]:fv[-1] + 1]
    if np.any(np.isnan(fhr)):
        nan_start = None
        segs = []
        for i in range(len(fhr)):
            if np.isnan(fhr[i]):
                if nan_start is None:
                    nan_start = i
            elif nan_start is not None:
                segs.append((nan_start, i - 1))
                nan_start = None
        if nan_start is not None:
            segs.append((nan_start, len(fhr) - 1))
        for s, e in segs:
            seg = e - s + 1
            vb = fhr[s - 1] if s > 0 else None
            va = fhr[e + 1] if e < len(fhr) - 1 else None
            if seg > config.MISSING_THRESHOLD:
                if vb is not None and va is not None:
                    fhr[s:e + 1] = np.linspace(vb, va, seg + 2)[1:-1]
                elif vb is not None:
                    fhr[s:e + 1] = vb
                elif va is not None:
                    fhr[s:e + 1] = va
                else:
                    fhr[s:e + 1] = 0
            else:
                fhr[s:e + 1] = np.nanmedian(fhr)
    fhr = np.nan_to_num(fhr, nan=0)
    if len(fhr) < config.MIN_SIGNAL_LENGTH:
        return None
    return fhr


def stage2_uc_fm(fhr_clean, uc_raw):
    if uc_raw is None or len(uc_raw) != len(fhr_clean):
        uc_clean = np.zeros_like(fhr_clean)
    else:
        uc_clean = uc_raw.copy()
        nan = np.isnan(uc_clean)
        if np.any(nan):
            v = uc_clean[~nan]
            uc_clean[nan] = np.median(v) if len(v) else 0
    fm_clean = fhr_clean - uc_clean
    return uc_clean, fm_clean


def stage3_baseline(fhr_clean, config):
    """SG滤波 + EMD去高频 + KMeans极值点基线; 失败回退滑动中位数"""
    if len(fhr_clean) < config.SG_WINDOW:
        base = np.full_like(fhr_clean, np.median(fhr_clean))
        return fhr_clean - base
    try:
        fhr_sg = savgol_filter(fhr_clean, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        fhr_sg = fhr_clean.copy()
    fhr_emd = fhr_sg.copy()
    try:
        imfs = PyEMD()(fhr_sg)
        if imfs.shape[0] > 1:
            nr = max(1, min(int(imfs.shape[0] * 0.3), int(imfs.shape[0] * config.EMD_MAX_IMFS_TO_REMOVE)))
            fhr_emd = np.sum(imfs[nr:], axis=0)
    except Exception:
        pass
    base = None
    try:
        peaks = [i for i in range(1, len(fhr_emd) - 1) if fhr_emd[i] > fhr_emd[i - 1] and fhr_emd[i] > fhr_emd[i + 1]]
        troughs = [i for i in range(1, len(fhr_emd) - 1) if fhr_emd[i] < fhr_emd[i - 1] and fhr_emd[i] < fhr_emd[i + 1]]
        ext = peaks + troughs
        if len(ext) >= config.KMEANS_N_CLUSTERS:
            vals = fhr_emd[ext].reshape(-1, 1)
            km = KMeans(n_clusters=config.KMEANS_N_CLUSTERS, random_state=0, n_init=10).fit(vals)
            centers = km.cluster_centers_.flatten()
            bi = [ext[i] for i in range(len(km.labels_)) if km.labels_[i] == np.argmin(centers)]
            if len(bi) >= 2:
                bi = sorted(bi)
                base = np.interp(np.arange(len(fhr_emd)), bi, fhr_emd[bi])
    except Exception:
        pass
    if base is None:
        w = max(31, len(fhr_emd) // 10) | 1
        try:
            base = median_filter(fhr_emd, size=w)
        except Exception:
            base = np.full_like(fhr_emd, np.median(fhr_emd))
    try:
        base = savgol_filter(base, config.SG_WINDOW, config.SG_POLYORDER)
    except Exception:
        pass
    std = fhr_emd - base
    if np.std(std) < 0.1:
        base = np.full_like(fhr_emd, np.median(fhr_emd))
        std = fhr_emd - base
    return std


def stage5_augment(fhr, uc, fm, label, config):
    L = len(fhr)
    T = config.TARGET_LENGTH
    out = []
    if L < T:
        return out
    if label == 0:
        s = max(0, L // 2 - T // 2)
        out.append({'fhr': fhr[s:s + T], 'uc': uc[s:s + T], 'fm': fm[s:s + T], 'label': label})
    else:
        if L == T:
            out.append({'fhr': fhr[:T], 'uc': uc[:T], 'fm': fm[:T], 'label': label})
        elif L < T + config.AUGMENT_SLIDE:
            out.append({'fhr': fhr[L - T:], 'uc': uc[L - T:], 'fm': fm[L - T:], 'label': label})
        else:
            n = min(1 + (L - T) // config.AUGMENT_SLIDE, 3)
            for k in range(n):
                s = k * config.AUGMENT_SLIDE
                if s + T <= L:
                    out.append({'fhr': fhr[s:s + T], 'uc': uc[s:s + T], 'fm': fm[s:s + T], 'label': label})
    return out


def stage6_unify(fhr, uc, fm, T):
    L = len(fhr)
    if L >= T:
        s = L // 2 - T // 2
        return fhr[s:s + T], uc[s:s + T], fm[s:s + T]
    p = T - L
    return np.pad(fhr, (0, p), mode='edge'), np.pad(uc, (0, p), mode='edge'), np.pad(fm, (0, p), mode='edge')


def stage7_split(fhr_all, uc_all, fm_all, labels_all, config):
    print('[阶段7] 分层划分')
    idx = np.arange(len(labels_all))
    itv, ite = train_test_split(idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=labels_all)
    itr, iva = train_test_split(itv, test_size=config.VAL_RATIO, random_state=config.RANDOM_STATE + 1, stratify=labels_all[itv])
    print(f'  训练 {len(itr)} / 验证 {len(iva)} / 测试 {len(ite)}')
    return {
        'train': {'fhr': fhr_all[itr], 'uc': uc_all[itr], 'fm': fm_all[itr], 'label': labels_all[itr]},
        'val':   {'fhr': fhr_all[iva], 'uc': uc_all[iva], 'fm': fm_all[iva], 'label': labels_all[iva]},
        'test':  {'fhr': fhr_all[ite], 'uc': uc_all[ite], 'fm': fm_all[ite], 'label': labels_all[ite]},
    }


def save_data(split, config):
    print('保存数据 ->', config.OUTPUT_PATH)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    for name, d in split.items():
        pd.DataFrame(d['fhr']).to_csv(os.path.join(config.OUTPUT_PATH, f'fhr_{name}.csv'), header=None, index=False)
        pd.DataFrame(d['uc']).to_csv(os.path.join(config.OUTPUT_PATH, f'uc_{name}.csv'), header=None, index=False)
        pd.DataFrame(d['fm']).to_csv(os.path.join(config.OUTPUT_PATH, f'fm_{name}.csv'), header=None, index=False)
        pd.DataFrame(d['label']).to_csv(os.path.join(config.OUTPUT_PATH, f'label_{name}.csv'), header=None, index=False)


def main():
    config = Config()
    t0 = time.time()
    print('产前 CTG 预处理 - 二分类')
    print(f'输入: {config.RAW_DATA_PATH}  输出: {config.OUTPUT_PATH}')

    pairs = collect_data_pairs(config)
    if not pairs:
        print('未找到有效数据对!')
        return

    all_fhr, all_uc, all_fm, all_lab = [], [], [], []
    processed = failed = skipped_q = 0
    for i, p in enumerate(pairs):
        if i % 500 == 0:
            print(f'  进度 {i}/{len(pairs)} (成功 {processed}, 失败 {failed}, Q剔除 {skipped_q})')
        fhr_raw, uc_raw, fm_raw = read_dat_file(p['dat_path'])
        if fhr_raw is None or len(fhr_raw) < 10:
            failed += 1
            continue
        if get_Q(fhr_raw, uc_raw) < config.Q_THRESHOLD:
            skipped_q += 1
            continue
        fhr_clean = stage1_fhr_cleaning(fhr_raw, config)
        if fhr_clean is None:
            failed += 1
            continue
        uc_clean, fm_clean = stage2_uc_fm(fhr_clean, uc_raw if len(uc_raw) == len(fhr_clean) else None)
        fhr_std = stage3_baseline(fhr_clean, config)
        for aug in stage5_augment(fhr_std, uc_clean, fm_clean, p['label'], config):
            f, u, m = stage6_unify(aug['fhr'], aug['uc'], aug['fm'], config.TARGET_LENGTH)
            all_fhr.append(f)
            all_uc.append(u)
            all_fm.append(m)
            all_lab.append(aug['label'])
        processed += 1

    print(f'\n处理完成: 原始 {len(pairs)}, 成功 {processed}, 失败 {failed}, Q剔除 {skipped_q}, 增强后 {len(all_lab)}')
    all_fhr = np.array(all_fhr)
    all_uc = np.array(all_uc)
    all_fm = np.array(all_fm)
    all_lab = np.array(all_lab)
    split = stage7_split(all_fhr, all_uc, all_fm, all_lab, config)
    save_data(split, config)
    print(f'预处理完成! 耗时 {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
