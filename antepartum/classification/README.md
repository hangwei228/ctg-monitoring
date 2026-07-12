# 多模态胎监分类（classification 模块）

产前 CTG 多模态 CNN 分类实现。三路 1D 残差卷积编码器（FHR / UC / FM）+ SE 注意力，拼接临床
stats 向量（孕周、年龄、CTG 指南特征）后接全连接分类头；信号前端含 **EVT-ADB** 去噪层。

## 输入格式

每个数据集目录下放置以下 CSV（**无表头**，每行一条记录）：

| 文件 | 内容 | 形状 |
|---|---|---|
| `fhrbs.csv` | 基线标准化 FHR `S=F−B` | `(N, 1125)` |
| `uc.csv` | 宫缩 | `(N, 1125)` |
| `fm.csv` | 胎动（二值） | `(N, 1125)` |
| `label.csv` | 整数标签 `0`/`1` 或 `0`/`1`/`2` | `(N,)` |
| `gest+age.csv` | 孕周、年龄 | `(N, 2)` |

原始 `.dat` 信号处理见仓库根 `preprocessing/`。

## 使用

```bash
pip install -r ../requirements.txt
python run.py --dataset all --task both          # center+mobile, 二分类+三分类
python run.py --dataset center --task binary      # 单子集/单任务
python run.py --dataset center --task binary --quick_test
```

| 参数 | 可选值 | 默认 | 说明 |
|---|---|---|---|
| `--dataset` | `center`/`mobile`/`all` | `center` | 数据集 |
| `--task` | `binary`/`three_class`/`both` | `both` | 任务 |
| `--result_dir` | 路径 | `./results` | 输出目录 |
| `--quick_test` | 开关 | 关 | 仅 5 epoch |

## 输出

`--result_dir` 下生成 `<dataset>_binary/`、`<dataset>_3class/`（各含 `best_model.keras`、
`final_model.keras`、ROC 与训练曲线 PNG）及 `all_results.txt` 指标汇总。

## 要点

- 信号长度 1125，逐样本 z-score 归一化；类别加权 + 噪声/mixup 增强 + L2=1e-5。
- 约 29.6 万参数，CPU 可训练。
- 三分类依赖真实"可疑"类样本，否则退化为二分类。
