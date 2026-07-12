"""
EVT-ADB: Adaptive Soft Thresholding Residual Denoising Block based on Extreme Value Theory
纯去噪模块，带运行时开关 (_soft_on / _evt_on)
"""
import keras
import keras.ops as K
from keras import layers


class EVT_ADB_v2(layers.Layer):
    """EVT-ADB 自适应软阈值残差去噪块

    包含三个功能块:
      1. 特征提取: Conv1D → BN → ReLU → Conv1D → BN
      2. 软阈值去噪: 通道注意力 → 自适应阈值 → 软阈值
      3. EVT去噪评估与动态补偿: 残差计算 → 超限判断 → 补偿

    运行时通过 _soft_on / _evt_on 控制软阈值和EVT的启闭。

    Args:
        f: 输出通道数
        k: 卷积核大小 (default: 5)
    """
    def __init__(self, f, k=5, **kw):
        super().__init__(**kw)
        self.f, self.k = f, k
        self._soft_on = True   # 软阈值开关
        self._evt_on = True    # EVT补偿开关

    def build(self, ins):
        self.c1 = layers.Conv1D(self.f, self.k, 1, 'same')
        self.c1.build(ins)
        self.b1 = layers.BatchNormalization()
        self.b1.build(ins)

        s = list(ins)
        s[2] = self.f
        self.c2 = layers.Conv1D(self.f, self.k, 1, 'same')
        self.c2.build(s)
        self.b2 = layers.BatchNormalization()
        self.b2.build(s)

        self.sc = layers.Dense(self.f, 'sigmoid')
        self.sc.build((ins[0], ins[2]))

        # EVT可学习参数
        self.th = self.add_weight(
            name='th', shape=(),
            initializer=keras.initializers.Constant(2.0),
            trainable=True)
        self.sv = self.add_weight(
            name='sv', shape=(),
            initializer=keras.initializers.Ones(),
            trainable=True)

    def call(self, x, training=None):
        r = x

        # ===== 功能块一: 特征提取 =====
        x = K.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))

        # ===== 功能块二: 自适应软阈值去噪 =====
        if self._soft_on:
            am = K.mean(K.abs(x), 1, keepdims=True)               # μ: 各通道绝对值均值
            s = K.reshape(self.sc(K.mean(x, 1)), (-1, 1, self.f)) # s: 可学习缩放系数
            thr = am * s                                           # τ: 各通道自适应阈值
            xd = K.sign(x) * K.maximum(K.abs(x) - thr, 0.0)       # 软阈值去噪
        else:
            xd = x

        # ===== 功能块三: EVT去噪评估与动态补偿 =====
        if self._evt_on:
            res = K.abs(r[..., :self.f] - xd[..., :self.f])            # R: 去噪残差
            exc = K.maximum(res - K.softplus(self.th), 0.0)            # E: 超限部分
            w = 1 - K.exp(-exc / (K.abs(self.sv) + 1e-6))              # W: 补偿权重
            xd = xd + w * res * 0.2                                     # 补偿后特征

        return xd
