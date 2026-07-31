# 双自适应软阈值多模态网络（DAST-MMNet）

本模型为面向产时胎儿监护，基于双自适应软阈值多模态网络的多模态胎儿健康状态分类框架。融合胎心率（FHR）、宫缩信号（UC）和临床特征，通过专属的特征提取与自适应去噪，跨模态深度融合网络，结合对比增强与重掩码机制，进行面向产时胎儿监护中的严重类别不平衡的正常/异常胎儿二分类。

## 目录结构

```
DAST-MMNet/
├── config.py                 # 超参数与路径配置
├── train.py                  # 训练主入口
├── test.py                   # 自定义 keras.Model 模板
├── requirements.txt          # 依赖
├── README.md
├── models/
│   ├── attention.py          # 注意力模块
│   ├── encoder.py            # 双流 1D 编码器 + 残差收缩块
│   └── training_model.py     # 训练模型
├── losses/
│   └── losses.py             # 损失函数
├── data/
│   ├── dataset.py            # 数据加载
│   └── generators.py         # 数据生成器（块掩码增强）
├── metrics/
│   └── evaluation.py         # 评估指标
└── utils/
    └── helpers.py            # 学习率调度、训练循环辅助
```

## 各模块说明

### `models/` — 模型架构

| 文件 | 内容 |
|------|------|
| `attention.py` | `se_block()`、`SEBlock`、`ChannelAttention`、`SpatialAttention`、`CBAMBlock`、`MS_CAM`、`AFF`（2D）、`AFF_1D`、`ChannelSpatialAttention`、`DualAttention` |
| `encoder.py` | 编码器相关：`SoftThresholdingLocal/Global/Output`（软阈值去噪层）、`AFF1D`（带阈值化的 AFF）、`ResidualShrinkageBlock1D`（残差收缩块）、`InvertedResidualBlock`、`Encoder`（双流编码器） |
| `training_model.py` | `AdaptiveWeighting`（自适应加权）、`TrainingModel`（端到端训练模型，含 `train_step`/`test_step`） |

### `losses/losses.py` — 损失函数

| 函数/类 | 说明 |
|---------|------|
| `NTXentLoss` | 对比学习损失（NT-Xent） |
| `sce_loss` | SiamCE 平滑交叉熵 |
| `binary_crossentropy_loss` | 二分类交叉熵 + 假阴性惩罚 |
| `dice_loss` | Dice 损失 |
| `bce_dice_loss` | BCE + Dice 组合损失 |
| `focal_loss` | Focal Loss |



### `metrics/evaluation.py` — 评估指标

| 类 | 方法 |
|----|------|
| `rec_pre` | `recall()`、`precision()` |
| `f1`(继承 `rec_pre`) | `f1_scores()` |
| `his_sum` | `score_sum()`（混淆矩阵、分类报告、ROC/AUC、Kappa、MCC）、`paint_history()`（训练曲线绘图） |


## 训练流程

### 1. 准备数据

在项目根目录下创建 `dataset_new/` 文件夹，放入 CSV 文件：

| 文件 | 说明 |
|------|------|
| `fhr_bs_uc_train.csv` | 训练信号，shape = (N, 12000)，前半 FHR 后半 UC |
| `fhr_bs_uc_val.csv` | 验证信号 |
| `fhr_bs_uc_test.csv` | 测试信号 |
| `train_c.csv` | 训练临床特征（14 列） |
| `val_c.csv` | 验证临床特征 |
| `test_c.csv` | 测试临床特征 |
| `y_train.csv` | 训练标签 |
| `y_val.csv` | 验证标签 |
| `y_test.csv` | 测试标签 |

数据路径在 `config.py` 中配置：
```python
DATA_ROOT = './dataset_new/'
```

### 2. 配置参数

编辑 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | `32` | 批次大小 |
| `EPOCH` | `200` | 训练轮数 |
| `MODE` | `'AFF'` | 融合模式：`'AFF'`（自适应融合）或 `'None'`（拼接） |
| `MODEL_NAME` | `'FHR+UC+Clinic+AFF+2025_AFF'` | 模型保存名 |
| `LR` | `3.5e-5` | 学习率 |
| `RESULT_DIR` | `'./Result/'` | 模型与日志输出目录 |
| `CUDA_VISIBLE_DEVICES` | `"0"` | 指定 GPU |

### 3. 运行训练

```bash
python train.py
```

流程：
1. **构建模型** — `TrainingModel`（Encoder + AFF 融合 + 分类头）
2. **加载数据** — 读取 CSV → reshape → one-hot 编码
3. **训练** — 每轮：块掩码增强 → 前向 → 计算 BCE loss → 反向传播 → 验证
4. **回调** — ModelCheckpoint（保存最佳）、ReduceLROnPlateau（factor=0.92, patience=2）、EarlyStopping（patience=50）、TensorBoard
5. **输出** — 保存模型权重、训练历史、指标曲线图，打印混淆矩阵/分类报告/ROC-AUC/Kappa/MCC

### 4. 输出文件

保存到 `./Result/`：

| 文件 | 说明 |
|------|------|
| `{timestamp}{model_name}_best.ckpt` | 验证 loss 最低的模型 |
| `{timestamp}{model_name}_model.ckpt` | 最终模型权重 |
| `{timestamp}{model_name}_history.txt` | 训练历史（pickle 格式） |
| `{timestamp}{model_name}.jpg` | loss 和 F1 曲线图 |

## 数据格式

- **FHR+UC 信号**: 读取后 reshape 为 `(N, 2, 6000, 1)`，通道 0 = FHR，通道 1 = UC
- **临床特征**: reshape 为 `(N, 1, 14)`，14 维临床指标
- **标签**: 二分类，`{0: 正常, 1: 异常}`，原始标签中 2（可疑）归为异常
- 标签经 `to_categorical` 转为 one-hot 编码

## 数据集介绍

测试使用公共CTU-UHB数据集，根据胎儿出生之后的脐动脉血ph值对样本进行分类，频率下采样到1Hz,二分类标签是ph>=7.15为正常，ph<7.15是1酸血症；三分类是ph>7.2/7.15-7.2/<7.15三类；
数据集官方地址：https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

使用请同时引用：

- Chudáček, V., Spilka, J., Burša, M., Janků, P., Hruban, L., Huptych, M., & Lhotská, L. (2014). Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*, 14, 16. https://doi.org/10.1186/1471-2393-14-16

## 模型架构
```

**RSB** = Residual Shrinkage Block（残差收缩块），包含：
- BN → ReLU → Conv1D → BN → ReLU → Conv1D
- 可选软阈值化：GAP → Dense → Sigmoid → 阈值缩放 → 符号收缩
- 跳跃连接 + 平均池化下采样

![模型架构图](3445e5e4455056202888d8f443035e9e-1.png)
'''
模型通过自适应去噪模块有效抑制宫缩压（UC）信号中的噪声干扰，
并利用多模态融合机制捕捉胎心率、宫缩压信号与母体临床数据间的非线性跨模态交互，
以实现全面、精准的产时胎儿评估。
同时，对比增强与重新掩码机制有效缓解了类别不平衡问题，强化了对关键特征的关注

'''

## 依赖

```
tensorflow>=2.10
keras>=2.10
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
scikit-learn>=1.0
```

## 引用

- **[产时 Intrapartum]** Yao, H., Lin, X., Tang, L., Liu, G., Chen, Q., & Wei, H.\* (2026). Dual adaptive soft thresholding multimodal networks with contrastive augmentation for intrapartum fetal monitoring. *Engineering Applications of Artificial Intelligence*, 170, 114179. https://doi.org/10.1016/j.engappai.2026.114179