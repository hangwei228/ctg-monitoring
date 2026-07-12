# 产前信号预处理（preprocessing 模块）

将原始 **MyCTG_T 格式 `.dat` 记录**（FHR/UC/FM @1Hz）转为分类模块可用的训练 CSV。流水线：

1. **数据加载**：遍历 `RAW_DATA_PATH` 各阶段 `Dat/` 目录，按编号匹配标签 CSV（有反应/无反应/可疑）与临床信息（孕周、年龄）。
2. **Q 值筛选**（阈值 0.6）：剔除低质量记录。
3. **FHR 异常/缺失处理**：异常值标记、首尾截断、缺失段插值或连接。
4. **UC/FM 同步**：长度对齐、缺失插值；`FM = FHR − UC`。
5. **基线提取**：SG 滤波 + EMD 分解 + KMeans 极值点聚类；回退滑动中位数。
6. **标准化**：`S = F − B`。
7. **数据增强**：非正态类滑窗截取多段。
8. **长度统一**：900 点（15 分钟）。
9. **分层划分**：训练/验证/测试（70/15/15）。

输出：`fhr_{train,val,test}.csv`、`uc_*.csv`、`fm_*.csv`、`label_*.csv`。

> 信号去噪由分类模型中的 **EVT-ADB** 模块完成，本脚本不做去噪。

## 文件

- `三分类预处理.py`：标签 有反应(0) / 无反应(1) / 可疑(2)。
- `二分类预处理.py`：无反应 + 可疑 合并为异常(1)。

## 使用

顶部 `Config` 配置路径后运行（默认输出到 `./output/...`，按需修改）：

```bash
cd preprocessing
pip install -r ../requirements.txt   # 需 PyEMD
python 三分类预处理.py
python 二分类预处理.py
```

> 注：分类模块读取的是已准备好的 `fhrbs.csv / uc.csv / fm.csv / label.csv / gest+age.csv`
> 目录格式；本流水线输出为 `fhr_{split}.csv` 切分格式。若直接喂给分类模块，请合并切分信号并
> 补齐 `gest+age.csv`，或改写 `classification/data_loader.py` 以读取切分格式。
