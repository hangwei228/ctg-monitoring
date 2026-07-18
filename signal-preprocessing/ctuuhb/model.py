import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Lambda
)


class SimpleEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.conv1 = Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')
        self.conv2 = Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu')
        self.conv3 = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        return x

    def get_config(self):
        config = super(SimpleEncoder, self).get_config()
        return config


def smooth_threshold(x, lambda_value):
    from tensorflow.keras.activations import softplus
    return softplus(x - lambda_value) - softplus(-x - lambda_value)


def custom_smooth_thresholding(x):
    return smooth_threshold(x[0], x[1])


class SoftThresholdingLocal(layers.Layer):
    def __init__(self, inter_channels, channels, **kwargs):
        super(SoftThresholdingLocal, self).__init__(**kwargs)
        self.inter_channels = inter_channels
        self.channels = channels
        self.conv1 = layers.Conv1D(channels, kernel_size=1, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense = layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        abs_mean = tf.expand_dims(tf.reduce_mean(tf.math.abs(x), axis=1), axis=1)
        scales = self.dense(self.relu(self.bn1(self.conv1(abs_mean))))
        thres = abs_mean * scales
        x_abs = tf.math.abs(x)
        n_sub = Lambda(custom_smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)

    def get_config(self):
        config = super(SoftThresholdingLocal, self).get_config()
        config.update({'inter_channels': self.inter_channels, 'channels': self.channels})
        return config


class SoftThresholdingGlobal(layers.Layer):
    def __init__(self, inter_channels, channels, **kwargs):
        super(SoftThresholdingGlobal, self).__init__(**kwargs)
        self.inter_channels = inter_channels
        self.channels = channels
        self.conv1 = layers.Conv1D(inter_channels, kernel_size=1, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense = layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        abs_mean = tf.expand_dims(tf.reduce_mean(tf.math.abs(x), axis=1), axis=1)
        scales = self.dense(self.relu(self.bn1(self.conv1(abs_mean))))
        thres = abs_mean * scales
        x_abs = tf.math.abs(x)
        n_sub = Lambda(custom_smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)

    def get_config(self):
        config = super(SoftThresholdingGlobal, self).get_config()
        config.update({'inter_channels': self.inter_channels, 'channels': self.channels})
        return config


class SoftThresholdingOutput(layers.Layer):
    def __init__(self, inter_channels, channels, **kwargs):
        super(SoftThresholdingOutput, self).__init__(**kwargs)
        self.inter_channels = inter_channels
        self.channels = channels
        self.conv1 = layers.Conv1D(inter_channels, kernel_size=1, strides=1)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense = layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        abs_mean = tf.expand_dims(tf.reduce_mean(tf.abs(x), axis=1), axis=1)
        scales = self.dense(self.relu(self.bn1(self.conv1(abs_mean))))
        thres = abs_mean * scales
        x_abs = tf.math.abs(x)
        n_sub = Lambda(custom_smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)

    def get_config(self):
        config = super(SoftThresholdingOutput, self).get_config()
        config.update({'inter_channels': self.inter_channels, 'channels': self.channels})
        return config


class AFF1D(layers.Layer):
    def __init__(self, channels, r, **kwargs):
        super(AFF1D, self).__init__(**kwargs)
        self.channels = channels
        self.r = r
        self.local_att = tf.keras.Sequential([
            layers.Conv1D(channels // r, kernel_size=1, strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(channels, kernel_size=1, strides=1),
            layers.BatchNormalization()
        ])
        self.global_att = tf.keras.Sequential([
            layers.Conv1D(channels // r, kernel_size=1, strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(channels, kernel_size=1, strides=1),
            layers.BatchNormalization()
        ])
        self.sigmoid = layers.Activation(tf.nn.sigmoid)
        self.local_thresholding = SoftThresholdingLocal(channels // r, channels)
        self.global_thresholding = SoftThresholdingGlobal(channels // r, channels)
        self.output_thresholding = SoftThresholdingOutput(channels // r, channels)

    def call(self, x, y):
        xy = x + y
        xl = self.local_att(xy)
        xg = self.global_att(xy)

        xl = self.local_thresholding(xl)
        xg = self.global_thresholding(xg)

        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + y * (1 - wei)

        xo_soft_thresholded = self.output_thresholding(xo)
        return xo_soft_thresholded

    def get_config(self):
        config = super(AFF1D, self).get_config()
        config.update({'channels': self.channels, 'r': self.r})
        return config


class AFFClassifier(Model):
    def __init__(self, **kwargs):
        super(AFFClassifier, self).__init__(**kwargs)

        self.encoder_fhr = SimpleEncoder()
        self.encoder_uc = SimpleEncoder()
        self.encoder_fm = SimpleEncoder()

        self.global_avg_pool = GlobalAveragePooling1D()

        self.aff_1d1 = AFF1D(128, 4)
        self.aff_1d2 = AFF1D(128, 4)
        self.aff_1d3 = AFF1D(128, 4)

        self.final_dense = Dense(2, activation='softmax')

    def get_config(self):
        config = super(AFFClassifier, self).get_config()
        return config

    def call(self, inputs, training=False, return_embedding=False):
        fhr, uc, fm = inputs

        fhr_feat = self.encoder_fhr(fhr)
        uc_feat = self.encoder_uc(uc)
        fm_feat = self.encoder_fm(fm)

        outputs1 = self.aff_1d1(fhr_feat, fm_feat)
        outputs2 = self.aff_1d2(uc_feat, fm_feat)
        outputs3 = self.aff_1d3(fhr_feat, uc_feat)

        outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
        outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
        outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)

        outputs = tf.concat([outputs1, outputs2, outputs3], axis=2)
        outputs = self.global_avg_pool(outputs)

        if return_embedding:
            return outputs

        outputs = self.final_dense(outputs)
        return outputs


def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        fl = alpha * y_true * tf.math.pow(1 - y_pred, gamma) * ce + \
             (1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma) * ce
        return tf.reduce_mean(fl)
    return loss_fn
