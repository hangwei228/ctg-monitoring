# CTG 胎儿健康分类 

基于 DAST-MMNet 下游分类模型，使用 FHR、UC、FM 三模态信号经 AFF1D 融合后进行胎儿健康状态二分类。

## 项目结构

```
.
├── model.py         # 模型定义 (SimpleEncoder, AFF1D, AFFClassifier)
├── data.py          # 数据加载 (相对路径)
├── config.py        # 配置文件 (路径、超参数)
├── train.py         # 训练脚本
├── api.py           # FastAPI 预测 API
├── requirements.txt
├── README.md
└── best_model.keras # 训练产出的权重文件
```

## 环境要求

- Python 3.9+
- CUDA 12.2 (如需 GPU 训练)
- 依赖：`pip install -r requirements.txt`

## 数据准备

数据文件位于项目上级目录的 `筛选_去噪结果_ctuuhb(1)/筛选_去噪结果_ctuuhb/ctuuhb_result_2/`，包含：

| 文件 | 说明 |
|------|------|
| `fhr_{split}.csv` | 胎心率 (900 点) |
| `uc_{split}.csv` | 宫缩压力 (900 点) |
| `fm_{split}.csv` | 胎动 (900 点) |
| `label_{split}.csv` | 标签 (0=正常, 1=异常) |

`{split}` 为 `train` / `val` / `test`。

如需修改数据路径，编辑 `config.py` 中的 `DATA_DIR`。

## 训练

```bash
python train.py
```

输出：best_model.keras、控制台打印测试集评估指标。


## API 服务

启动：

```bash
python api.py
```

服务监听 `http://0.0.0.0:8000`

### API 端点

**GET /** — 服务信息

**GET /health** — 健康检查

**POST /predict** — 预测 (multipart/form-data)

请求参数 (三个 CSV 文件)：

| 字段 | 类型 | 说明 |
|------|------|------|
| `fhr` | file | 胎心率 CSV (1×900, 无表头) |
| `uc` | file | 宫缩压力 CSV (1×900, 无表头) |
| `fm` | file | 胎动 CSV (1×900, 无表头) |

响应：

```json
{
  "label": 0,
  "class_name": "正常",
  "probability_normal": 0.81,
  "probability_abnormal": 0.19
}
```

### 调用示例

```bash
curl -X POST http://localhost:8000/predict \
  -F "fhr=@fhr_test_0.csv" \
  -F "uc=@uc_test_0.csv" \
  -F "fm=@fm_test_0.csv"
```

