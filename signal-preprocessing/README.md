# Signal Preprocessing 信号预处理

本目录包含 CTG 胎心监护信号的预处理代码

## 代码组织

按数据集分为两个子目录：

- `ctuuhb/`：针对 **CTU-UHB 公开数据集**（PhysioNet）的预处理代码，含二分类与三分类版本
- `private_dataset/`：针对**私有数据集**的预处理代码，含二分类与三分类版本

## 处理流程

两套代码使用相同的核心方法，差异在数据格式与标签解析：

1. Q 值筛选（阈值 0.6）
2. 去噪
3. FHR / UC 信号清洗
4. SG 滤波 + EMD + K-Means 基线提取
5. 标准化：S(t) = F(t) - B(t)
6. 滑动窗口数据增强
7. 长度统一为 900 点
8. 70 / 15 / 15 分层划分（train / val / test）

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

具体使用方式与数据准备请参考各子目录下的 README。
