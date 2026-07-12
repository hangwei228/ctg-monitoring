# Quality 

信号质量评估工具，计算 FHR/UC 信号的 Q 值，用于筛选低质量信号

## Q值公式

$$Q = 1 - \frac{\sum X_n \log_{10} X_n + \sum Y_m \log_{10} Y_m}{2 \cdot l \cdot \log_{10} l}$$

- Xₙ: FHR 缺失段长度（FHR < 50 或 > 200 的连续段）
- Yₘ: UC 缺失段长度（连续 15 点以上变化 ≤ 1 的段）
- l: 信号总长度

Q 值范围 [0, 1]，越高表示信号质量越好。默认筛选阈值 Q < 0.6 判定为不合格。

## 输入

| 文件 | 格式 | 说明 |
|------|------|------|
| `fhr_train.csv` | CSV, header=None, 每行一个样本 | 原始 FHR 信号 |
| `uc_train.csv` | CSV, header=None, 每行一个样本 | 原始 UC 信号 |

## 输出

```
共 N 个样本
Begin...
0.csv
1.csv
...
finish!
[Q<0.6 的样本索引列表]
```

## 使用

```bash
python Quality.py
```

需与输入 CSV 文件在同一目录下运行。
