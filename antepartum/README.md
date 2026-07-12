# 多模态胎监信号分类（产前 CTG 自动判读）

基于多模态 1D 残差 CNN 的**产前（antepartum）胎心监护（CTG）信号自动判读**实现。
输入为三路时序信号（FHR / UC / FM），输出为正常/异常（二分类）或正常/可疑/病理性（三分类）。

信号去噪采用 **EVT-ADB** 自适应软阈值残差去噪模块（作为模型前端层）。

## 目录结构

```
.
├── classification/        # 多模态 CNN 分类（核心代码）
│   ├── model.py           # 三路 1D 残差 CNN + SE 注意力 + EVT-ADB 去噪 + 临床特征融合
│   ├── evt_adb.py         # EVT-ADB 去噪模块
│   ├── data_loader.py     # 数据加载、CTG 临床特征、分层划分、归一化
│   ├── run.py             # 训练 + 评估入口
│   └── README.md
├── preprocessing/         # 原始 .dat 信号预处理流水线
│   ├── 三分类预处理.py     # Q值筛选 + 清洗 + 基线提取 + 标准化 + 增强（三分类）
│   ├── 二分类预处理.py     # 同上，无反应+可疑合并为异常
│   └── README.md
├── report.md              # 实验结果报告
├── requirements.txt
└── README.md
```

## 实验结果（私有数据集）

数据为私有产前 CTG 数据集（中心站 / 移动式两个子集），**不对外公开**。在自有数据上的结果：

| 数据集 | 任务 | Accuracy | AUC | Kappa |
|---|---|---|---|---|
| center | 二分类 | 0.8688 | 0.9397 | 0.7376 |
| center | 三分类 | 0.8847 | 0.9487 | 0.7694 |
| mobile | 二分类 | 0.6723 | 0.7945 | — |
| mobile | 三分类 | 0.7921 | 0.8806 | 0.5276 |

> 详细说明见 [`report.md`](report.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 分类：在 center + mobile 上跑 二分类 + 三分类
cd classification
python run.py --dataset all --task both

# 预处理：原始 .dat -> 训练用 CSV
cd ../preprocessing
python 三分类预处理.py
```

## 依赖

Python 3.8 / 3.10；TensorFlow ≥ 2.13（CPU 即可）；numpy / pandas / scikit-learn / scipy / matplotlib；
预处理额外需要 `PyEMD`。

## 数据说明

原始信号与标签受数据使用协议限制，**不随本仓库发布**。请按 `classification/README.md` 中的格式自备数据目录。
模型检查点与大规模中间文件由运行脚本自动生成，不纳入版本控制（见 `.gitignore`）。
