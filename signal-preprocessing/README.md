# Signal Preprocessing 信号预处理

本目录包含 CTG 胎心监护信号的预处理代码。

## 代码组织

分为两个部分：

- `ctuuhb/`：使用 **CTU-UHB 公开数据集**（PhysioNet）进行测试的**预处理流程**，拥有可**独立调用的功能模块**，是完整流程中核心算法的拆解
- `private_dataset/`：针对**私有数据集**的**完整预处理流程**，含二分类版本

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


 ## 参数说明与调整建议 Parameter Configuration

本项目参数在自建数据集上调优得到。**如需应用到其他数据集，请根据实际信号质量、采样率、数据规模自行调整。**

### 参数位置

具体参数在各 py 文件的开头部分定义，修改时请**同时保持二分类和三分类版本的参数一致**，以避免结果偏差。



## 引用与相关成果 Citation & Related Achievements

如果本项目对你的研究有帮助，请引用以下：
### 论文 Paper

- Fei, Y., Chen, F., He, L., Chen, J., Hao, Y., Li, X., Liu, G., Chen, Q., Li, L., & Wei, H.\* (2022). Intelligent classification of antenatal cardiotocography signals via multimodal bidirectional gated recurrent units. *Biomedical Signal Processing and Control*, 78, 104008.

  ### 数据集引用 Dataset

本项目 `signal-preprocessing/ctuuhb/` 模块使用了 CTU-UHB Intrapartum Cardiotocography Database（PhysioNet 公开数据集）。如使用请同时引用：

- Chudáček, V., Spilka, J., Burša, M., Janků, P., Hruban, L., Huptych, M., & Lhotská, L. (2014). Open access intrapartum CTG database. *BMC Pregnancy and Childbirth*, 14, 16. https://doi.org/10.1186/1471-2393-14-16

数据集官方地址：https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
