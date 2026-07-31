import tensorflow as tf
from tensorflow.keras import layers
from losses.losses import sce_loss
from functools import partial


def abs_backend(inputs):
    return tf.keras.backend.abs(inputs)


def expand_dim_backend(inputs):
    return tf.keras.backend.expand_dims(inputs, 1)


def sign_backend(inputs):
    return tf.keras.backend.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = tf.keras.backend.expand_dims(inputs, -1)
    inputs = tf.keras.backend.spatial_2d_padding(inputs, ((0, 0), (pad_dim, pad_dim)), 'channels_last')
    return tf.keras.backend.squeeze(inputs, -1)


def soft_threshold(x, lambda_value):
    from tensorflow.keras.activations import softplus
    return softplus(x - lambda_value) - softplus(-x - lambda_value)


def smooth_thresholding(x):
    return soft_threshold(x[0], x[1])


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
        n_sub = layers.Lambda(smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)


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
        n_sub = layers.Lambda(smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)


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
        n_sub = layers.Lambda(smooth_thresholding)([x_abs, thres])
        return tf.multiply(tf.sign(x), n_sub)


class AFF1D(layers.Layer):
    def __init__(self, channels, r, **kwargs):
        super(AFF1D, self).__init__(**kwargs)
        self.channels = channels
        self.r = r
        self.local_att = tf.keras.Sequential([
            layers.Conv1D(channels // r, kernel_size=1, strides=1),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(channels, kernel_size=1, strides=1),
            layers.BatchNormalization(),
        ])
        self.global_att = tf.keras.Sequential([
            layers.Conv1D(channels // r, kernel_size=1, strides=1),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv1D(channels, kernel_size=1, strides=1),
            layers.BatchNormalization(),
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


class InvertedResidualBlock(layers.Layer):
    def __init__(self, inp, oup, expand_ratio, **kwargs):
        super(InvertedResidualBlock, self).__init__(**kwargs)
        hidden_dim = int(inp * expand_ratio)
        self.conv1 = layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.relu1 = layers.ReLU(max_value=6.0)
        self.pad = layers.ZeroPadding1D(padding=1)
        self.dw_conv = layers.DepthwiseConv1D(3, use_bias=False, depth_multiplier=hidden_dim)
        self.relu2 = layers.ReLU(max_value=6.0)
        self.conv2 = layers.Conv1D(oup, 1, use_bias=False)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pad(x)
        x = self.dw_conv(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class ResidualShrinkageBlock1D(layers.Layer):
    def __init__(self, nb_blocks, out_channels, downsample=False, downsample_strides=2, softthres=False, **kwargs):
        super(ResidualShrinkageBlock1D, self).__init__(**kwargs)
        self.nb_blocks = nb_blocks
        self.out_channels = out_channels
        self.downsample = downsample
        self.downsample_strides = downsample_strides
        self.softthres = softthres
        self.bn_layers = [layers.BatchNormalization() for _ in range(2 * nb_blocks)]
        self.relu_layers = [layers.Activation('relu') for _ in range(2 * nb_blocks)]
        self.conv_layers = [
            layers.Conv1D(out_channels, 3, strides=downsample_strides if i % 2 == 0 else 1,
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            for i in range(2 * nb_blocks)
        ]
        if softthres:
            self.gap = layers.GlobalAveragePooling1D()
            self.dense1 = layers.Dense(out_channels // 4, activation=None,
                                       kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            self.dense2 = layers.Dense(out_channels, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            self.bn_scale = layers.BatchNormalization()
            self.relu_scale = layers.Activation('relu')

    def call(self, incoming, training=False):
        residual = incoming
        in_channels = incoming.shape[-1]
        for i in range(self.nb_blocks):
            identity = residual
            downsample_strides = self.downsample_strides if self.downsample else 1
            residual = self.bn_layers[2 * i](residual, training=training)
            residual = self.relu_layers[2 * i](residual)
            residual = self.conv_layers[2 * i](residual)
            residual = self.bn_layers[2 * i + 1](residual, training=training)
            residual = self.relu_layers[2 * i + 1](residual)
            residual = self.conv_layers[2 * i + 1](residual)
            if self.softthres:
                residual_abs = tf.abs(residual)
                abs_mean = self.gap(residual_abs)
                scales = self.dense1(abs_mean)
                scales = self.bn_scale(scales, training=training)
                scales = self.relu_scale(scales)
                scales = self.dense2(scales)
                thres = scales * abs_mean
                thres = tf.expand_dims(thres, axis=1)
                sub = tf.maximum(residual_abs - thres, 0.0)
                residual = tf.sign(residual) * sub
            if downsample_strides > 1:
                identity = layers.AveragePooling1D(pool_size=1, strides=downsample_strides)(identity)
            if in_channels != self.out_channels:
                padding = self.out_channels - in_channels
                identity = tf.pad(identity, [[0, 0], [0, 0], [0, padding]])
            residual = layers.add([residual, identity])
        return residual


class Encoder(tf.keras.Model):
    def __init__(self, mode='soft_thresholding'):
        super(Encoder, self).__init__()
        dense_list = [128, 256, 396, 336, 256, 196, 128]
        self.dense_layers = [layers.Dense(i, activation='relu') for i in dense_list]
        self.conv1d1 = layers.Conv1D(128, kernel_size=14, strides=1, padding='same')
        self.conv1d2 = layers.Conv1D(128, kernel_size=7, strides=1, padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.global_avg_pool = layers.GlobalAveragePooling1D()

        if mode == 'conv':
            self.res_layers1 = [
                [layers.Conv1D(196, 3, strides=2 if i % 2 == 0 else 1, padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))
                 for i in range(2 * 4)]
                for j in range(4)
            ]
            self.res_layers2 = [
                [layers.Conv1D(128, 3, strides=2 if i % 2 == 0 else 1, padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))
                 for i in range(2 * 4)]
                for j in range(4)
            ]
        else:
            self.res_layers1 = [
                ResidualShrinkageBlock1D(1, 196, downsample=True, downsample_strides=2, softthres=False)
                for i in range(4)
            ]
            self.res_layers2 = [
                ResidualShrinkageBlock1D(1, 128, downsample=True, downsample_strides=2,
                                         softthres=True if i == 0 else False)
                for i in range(4)
            ]

        self.criterion = partial(sce_loss, alpha=1)

    def call(self, xi1, xi2, ci, training=True):
        xi1 = self.conv1d1(xi1)
        xi1 = tf.squeeze(xi1, axis=1)
        for layer in self.res_layers1:
            xi1 = layer(xi1)
        xi2 = self.conv1d2(xi2)
        xi2 = tf.squeeze(xi2, axis=1)
        for layer in self.res_layers2:
            xi2 = layer(xi2)
        for layer in self.dense_layers:
            ci = layer(ci)
        return xi1, xi2, ci