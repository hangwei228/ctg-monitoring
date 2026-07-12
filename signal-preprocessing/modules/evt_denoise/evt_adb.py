import keras
import keras.ops as K
from keras import layers


class EVT_ADB_v2(layers.Layer):
    """
    EVT-ADB: Adaptive Soft Thresholding Residual Denoising Block

    Three sub-blocks:
      1. Feature Extraction
      2. Soft Thresholding
      3. EVT Compensation

    """
    def __init__(self, f, k=5, **kw):
        super().__init__(**kw)
        self.f, self.k = f, k
        self._soft_on = True   
        self._evt_on = True    

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
        # Feature Extraction
        x = K.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        
        # Adaptive Soft Thresholding
        if self._soft_on:
            am = K.mean(K.abs(x), 1, keepdims=True)               
            s = K.reshape(self.sc(K.mean(x, 1)), (-1, 1, self.f)) 
            thr = am * s                                           
            xd = K.sign(x) * K.maximum(K.abs(x) - thr, 0.0)       
        else:
            xd = x

        # EVT Residual Compensatio
        if self._evt_on:
            res = K.abs(r[..., :self.f] - xd[..., :self.f])            
            exc = K.maximum(res - K.softplus(self.th), 0.0)            
            w = 1 - K.exp(-exc / (K.abs(self.sv) + 1e-6))              
            xd = xd + w * res * 0.2                                     

        return xd
