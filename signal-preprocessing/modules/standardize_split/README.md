# DataDivA

将原始 FHR/UC/FM 信号及基线数据按 70/15/15 比例分层划分训练集、验证集、测试集

## 输入

所有文件置于运行目录下：

| 文件 | 说明 |
|------|------|
| `fhr_train.csv` | 原始 FHR 信号（每行一个样本） |
| `uc_train.csv`  | UC 信号 |
| `fm_train.csv`  | FM 信号 |
| `fhrbs_train.csv` | 基线 B(t) |
| `label_train.csv` | 标签 (0/1/2) |

## 处理流程

1. 读取 5 个 CSV 文件
2. 计算标准化信号：`S(t) = FHR(t) - B(t)` → 保存为 `fhrbs`
3. 首次划分：70% 训练+验证 / 30% 测试（分层，`random_state=42`）
4. 二次划分：训练集中 80/20 分为最终训练集 / 验证集（`random_state=12`）
5. 保存 15 个 CSV 文件

## 输出 (`./output/`)

| 文件 | 划分 |
|------|------|
| `A_fhr_train.csv`, `A_uc_train.csv`, `A_fm_train.csv`, `A_fhrbs_train.csv`, `A_label_train.csv` | 训练集 (≈56%) |
| `A_fhr_val.csv`, `A_uc_val.csv`, `A_fm_val.csv`, `A_fhrbs_val.csv`, `A_label_val.csv` | 验证集 (≈14%) |
| `A_fhr_test.csv`, `A_uc_test.csv`, `A_fm_test.csv`, `A_fhrbs_test.csv`, `A_label_test.csv` | 测试集 (≈30%) |

每行对应一个样本，列对应时间序列各点。

## 使用

```bash
python datadivA.py
```

输出将保存在自动创建的 `./output/` 目录下。
