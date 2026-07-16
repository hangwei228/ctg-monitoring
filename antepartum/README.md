# 多模态胎监信号分类（产前 CTG 自动判读）

基于多模态 1D 残差 CNN 的**产前（antepartum）胎心监护（CTG）信号自动判读**实现。
输入为三路时序信号（FHR / UC / FM），输出为正常/异常（二分类）或正常/可疑/病理性（三分类）。


## 目录结构

```
.
├── classification/        # 多模态 CNN 分类（核心代码）
│   ├── model.py           # 三路 1D 残差 CNN + SE 注意力 + EVT-ADB 去噪 + 临床特征融合
│   ├── data_loader.py     # 数据加载、CTG 临床特征、分层划分、归一化
│   ├── run.py             # 训练 + 评估入口
│   └── README.md
├── preprocessing/         # 原始 .dat 信号预处理流水线
│   ├── 三分类预处理.py     # Q值筛选 + 清洗 + 基线提取 + 标准化 + 增强（三分类）
│   ├── 二分类预处理.py     # 同上，无反应+可疑合并为异常
│   └── README.md
├── requirements.txt
└── README.md
```

## 实验结果（私有数据集）

数据为私有产前 CTG 数据集，**不对外公开**。在自有数据上的结果：
经过判读与筛选后，中央站共有16355个案例，包括11998例正常类，4326例可疑类，31例异常类;


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

## 参数说明与调整建议 Parameter Configuration

本项目参数在自建数据集上调优得到。**如需应用到其他数据集，请根据实际信号质量、采样率、数据规模自行调整。**

## 标签标注方法 Label Annotation

本数据集的分类标签依据以下规范生成：

- **判读标准**：《妇产科学》（第 9 版，人民卫生出版社）中关于电子胎心监护（EFM）图形判读的相关章节
- **判读流程**：由 **3 名产科医师独立判读**，仅采用 **三人判读结果一致** 的样本纳入数据集
- **分类粒度**：
  - 二分类：正常 / 异常
  - 三分类：正常 / 可疑 / 异常

未达成三人一致的样本不纳入训练数据，以保证标签质量。

## 引用与相关成果 Citation & Related Achievements

如果本项目对你的研究有帮助，请引用以下：

### 专利 Patent

- 魏航, 陈帆, 费悦, 陈沁群, 洪佳明, 洪乐雄, 李丽, 林伙旺, 陈剑梅. 一种产前胎心监护信号智能判读方法. 中国发明专利, 专利号: ZL 202110798011.9, 授权公告日: 2024-03-15, 证书号: 第 6912221 号. 申请人: 广州三瑞医疗器械有限公司.

