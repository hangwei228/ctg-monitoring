# Signal Preprocessing 信号预处理

本目录包含 CTG 胎心监护信号的预处理代码。

## 代码组织

分为三个部分：

- `ctuuhb/`：针对 **CTU-UHB 公开数据集**（PhysioNet）的**完整预处理流程**，含二分类与三分类版本
- `private_dataset/`：针对**私有数据集**的**完整预处理流程**，含二分类版本
- `modules/`：可**独立调用的功能模块**，是完整流程中核心算法的拆解，公共 / 私有数据集共用

## 处理流程

两套完整脚本使用相同的核心方法，差异在数据格式与标签解析：

1. Q 值筛选（阈值 0.6）
2. FHR / UC 信号清洗
3. SG 滤波 + EMD + K-Means 基线提取
4. 标准化：S(t) = F(t) - B(t)
5. 滑动窗口数据增强
6. 长度统一为 900 点
7. 70 / 15 / 15 分层划分（train / val / test）

## 依赖环境

```
numpy
pandas
scipy
scikit-learn
PyEMD
wfdb   # 仅 ctuuhb/ 需要
```

## 使用说明

- 完整预处理 → 进 `ctuuhb/` 或 `private_dataset/`，看里面的 README
- 单独用某个算法模块（比如只做去噪） → 进 `modules/`，看对应子目录的 README
