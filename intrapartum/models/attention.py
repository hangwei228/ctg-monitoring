import tensorflow as tf
from tensorflow.keras import layers


def se_block(residual, name, ratio=8):
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.keras.initializers.Constant(value=0.0)
    channel = residual.shape[-1]
    squeeze = tf.reduce_mean(residual, axis=[1], keepdims=True)
    excitation = layers.Dense(
        units=channel // ratio, activation=tf.nn.relu,
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
        name=f'{name}_bottleneck_fc',
    )(squeeze)
    excitation = layers.Dense(
        units=channel, activation=tf.nn.sigmoid,
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
        name=f'{name}_recover_fc',
    )(excitation)
    return residual * excitation


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, name=None, **kwargs):
        super(SEBlock, self).__init__(name=name, **kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.dense1 = layers.Dense(
            units=channel // self.ratio, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_bottleneck_fc',
        )
        self.dense2 = layers.Dense(
            units=channel, activation=tf.nn.sigmoid,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_recover_fc',
        )

    def call(self, inputs):
        squeeze = tf.reduce_mean(inputs, axis=[1], keepdims=True)
        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)
        return inputs * excitation


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, name=None, **kwargs):
        super(ChannelAttention, self).__init__(name=name, **kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.avg_dense1 = layers.Dense(
            units=channel // self.ratio, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_mlp_0_avg',
        )
        self.avg_dense2 = layers.Dense(
            units=channel,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_mlp_1_avg',
        )
        self.max_dense1 = layers.Dense(
            units=channel // self.ratio, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_mlp_0_max',
        )
        self.max_dense2 = layers.Dense(
            units=channel,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            bias_initializer=tf.keras.initializers.Constant(value=0.0),
            name=f'{self.name}_mlp_1_max',
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1], keepdims=True)
        avg_branch = self.avg_dense2(self.avg_dense1(avg_pool))
        max_branch = self.max_dense2(self.max_dense1(max_pool))
        scale = tf.nn.sigmoid(avg_branch + max_branch)
        return inputs * scale


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, name=None, **kwargs):
        super(SpatialAttention, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            filters=1, kernel_size=self.kernel_size, strides=1,
            padding="same", activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            use_bias=False, name=f'{self.name}_conv',
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[2], keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=2)
        attention = self.conv(concat)
        return inputs * tf.nn.sigmoid(attention)


class CBAMBlock(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, name=None, **kwargs):
        super(CBAMBlock, self).__init__(name=name, **kwargs)
        self.channel_attention = ChannelAttention(ratio=ratio, name=f'{name}_ch_at')
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size, name=f'{name}_sp_at')

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


class MS_CAM(tf.keras.Model):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = tf.keras.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
        ])
        self.global_att = tf.keras.Sequential([
            layers.GlobalAvgPool2D(),
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
        ])
        self.sigmoid = layers.Activation(tf.nn.sigmoid)

    def call(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = tf.add(xl, xg)
        wei = self.sigmoid(xlg)
        return tf.multiply(x, wei)


class AFF(tf.keras.Model):
    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = tf.keras.Sequential([
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
        ])
        self.global_att = tf.keras.Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            layers.BatchNormalization(),
        ])
        self.sigmoid = layers.Activation(tf.nn.sigmoid)

    def call(self, x, y):
        xy = tf.add(x, y)
        xl = self.local_att(xy)
        xg = self.global_att(xy)
        xlg = tf.add(xl, xg)
        wei = self.sigmoid(xlg)
        return tf.add(tf.multiply(x, wei), tf.multiply(y, (1 - wei)))


class AFF_1D(tf.keras.Model):
    def __init__(self, channels=128, r=4):
        super(AFF_1D, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = tf.keras.Sequential([
            layers.Dense(inter_channels), layers.BatchNormalization(), layers.ReLU(),
            layers.Dense(channels), layers.BatchNormalization(),
        ])
        self.global_att = tf.keras.Sequential([
            layers.GlobalAveragePooling1D(keepdims=True),
            layers.Dense(inter_channels), layers.BatchNormalization(), layers.ReLU(),
            layers.Dense(channels), layers.BatchNormalization(),
        ])
        self.sigmoid = layers.Activation(tf.nn.sigmoid)

    def call(self, x, y):
        xy = x + y
        xl = self.local_att(xy)
        xg = self.global_att(xy)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei + y * (1 - wei)

    def get_config(self):
        config = super(AFF_1D, self).get_config()
        return config


class ChannelSpatialAttention(layers.Layer):
    def __init__(self, ratio=16):
        super(ChannelSpatialAttention, self).__init__()
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(1, kernel_size=1, strides=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(self.ratio, kernel_size=1, strides=1, use_bias=False)
        self.conv3 = layers.Conv2D(1, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.sigmoid = layers.Activation(tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        out = layers.Concatenate()([avg_out, max_out])
        out = layers.Reshape((-1, 1, 2))(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = layers.Reshape((-1, 1, 1))(out)
        out = self.sigmoid(out)
        return inputs * out

    def get_config(self):
        config = super(ChannelSpatialAttention, self).get_config()
        return config


class DualAttention(layers.Layer):
    def __init__(self, channels, ratio=16):
        super(DualAttention, self).__init__()
        self.channels = channels
        self.conv_channels = layers.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        self.bn_channels = layers.BatchNormalization()
        self.relu_channels = layers.ReLU()
        self.conv_spatial = layers.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        self.bn_spatial = layers.BatchNormalization()
        self.relu_spatial = layers.ReLU()
        self.channel_attention = ChannelSpatialAttention(ratio)
        self.spatial_attention = layers.Conv2D(1, kernel_size=1, strides=1, use_bias=False)
        self.sigmoid = layers.Activation(tf.nn.sigmoid)

    def call(self, inputs):
        out_channels = self.relu_channels(self.bn_channels(self.conv_channels(inputs)))
        out_spatial = self.relu_spatial(self.bn_spatial(self.conv_spatial(inputs)))
        out_spatial = self.sigmoid(self.spatial_attention(out_spatial))
        out = out_channels * out_spatial
        out = self.channel_attention(out)
        return out + inputs

    def get_config(self):
        config = super(DualAttention, self).get_config()
        return config