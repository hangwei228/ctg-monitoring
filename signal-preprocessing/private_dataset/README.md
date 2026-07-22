# 私有数据集预处理

## 数据说明

本代码用于处理采集的私有胎监数据。

**由于隐私保护，原始数据、中间结果、任何样本文件均不上传至公开仓库。** 

数据格式：`.dat` 信号文件 + CSV 标签文件，采样率 1Hz。

## 文件说明

- `binary_preprocess.py` — 二分类预处理
  - 标签规则：有反应 → 0（正常），无反应 + 可疑 → 1（合并为异常）

## 使用方法

1. 在本目录下新建 `data/raw/` 文件夹，把私有数据（`.dat` + CSV 标签）放进去
2. 运行：

```bash
python binary_preprocess.py

```

结果会输出到 `./data/output/binary_result/` 。

## 目录结构（运行后）

```
private_dataset/
├── README.md
├── binary_preprocess.py
└── data/                       # 使用者自建，不上传
    ├── raw/                    # 私有原始数据
    └── output/                 # 运行后自动生成
```
