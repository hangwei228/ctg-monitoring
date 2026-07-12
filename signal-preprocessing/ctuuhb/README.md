# CTU-UHB 公开数据集预处理

## 数据来源

CTU-UHB Intrapartum Cardiotocography Database，PhysioNet 免费下载：
https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

数据格式：WFDB 格式（`.dat` + `.hea`），原始 4Hz，代码内会下采样到 1Hz。

## 文件说明

- `binary_preprocess_ctuuhb.py` — 二分类预处理
  - 标签规则：pH ≥ 7.15 → 0（正常），pH < 7.15 → 1（酸血症）
- `three_class_preprocess_ctuuhb.py` — 三分类预处理
  - 标签规则：pH > 7.2 → 0（正常），7.15 ≤ pH ≤ 7.2 → 1（临界），pH < 7.15 → 2（酸血症）

## 使用方法

1. 从上述 PhysioNet 链接下载数据集
2. 在本目录下新建 `data/raw/` 文件夹，把解压后的所有 `.dat` 和 `.hea` 文件放进去
3. 运行：

```bash
python binary_preprocess_ctuuhb.py
# 或
python three_class_preprocess_ctuuhb.py
```

结果会输出到 `./data/output/binary_result/` 或 `./data/output/three_class_result/`。

## 目录结构（运行后）

```
ctuuhb/
├── README.md
├── binary_preprocess_ctuuhb.py
├── three_class_preprocess_ctuuhb.py
├── denoised_result_ctuuhb.zip
└── data/                       # 使用者自建
    ├── raw/                    # PhysioNet 下载的数据
    └── output/                 # 运行后自动生成
```
