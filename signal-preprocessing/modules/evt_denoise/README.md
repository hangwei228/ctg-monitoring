# EVT-ADB 

`EVT_ADB_v2` 是一个 Keras 自定义层，可作为 Conv1D 网络中的即插即用去噪模块。

## 架构

```
输入 x
  │
  ├─ 功能块一: 特征提取
  │   Conv1D → BN → ReLU → Conv1D → BN
  │
  ├─ 功能块二: 自适应软阈值去噪 (可开关 _soft_on)
  │   通道注意力 → 自适应阈值 τ → 软阈值
  │   τ = μ · s    (μ: 绝对值均值, s: 可学习缩放)
  │
  └─ 功能块三: EVT 动态补偿 (可开关 _evt_on)
       残差 R → 超限 E → 补偿权重 W → 补偿
       W = 1 - exp(-E / |sv|)
```

## 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `f` | int | 输出通道数 |
| `k` | int, default=5 | 卷积核大小 |

## 运行时开关

```python
layer._soft_on = True   # 软阈值去噪
layer._evt_on  = True   # EVT 残差补偿
```

## 使用示例

```python
from evt_adb import EVT_ADB_v2

inputs = keras.Input((timesteps, channels))
x = EVT_ADB_v2(f=64)(inputs)
```

## 依赖

- Keras 3
