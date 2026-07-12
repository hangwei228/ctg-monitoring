# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow.keras.regularizers import l2

from evt_adb import EVT_ADB_v2


def se_block(x, ratio=8):
    channels = x.shape[-1]
    se = kl.GlobalAveragePooling1D()(x)
    se = kl.Dense(max(channels // ratio, 4), activation='relu')(se)
    se = kl.Dense(channels, activation='sigmoid')(se)
    return kl.Multiply()([x, se])


def residual_block(x, filters, kernel_size=3, stride=1, pool_size=2, dropout_rate=0.25, use_se=True, l2_reg=1e-4):
    shortcut = x
    out = kl.Conv1D(filters, kernel_size, strides=stride, padding='same',
                    kernel_regularizer=l2(l2_reg))(x)
    out = kl.BatchNormalization()(out)
    out = kl.Activation('relu')(out)
    out = kl.Conv1D(filters, kernel_size, padding='same',
                    kernel_regularizer=l2(l2_reg))(out)
    out = kl.BatchNormalization()(out)
    if shortcut.shape[-1] != filters or stride > 1:
        shortcut = kl.Conv1D(filters, 1, strides=stride, padding='same',
                             kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = kl.BatchNormalization()(shortcut)
    out = kl.Add()([out, shortcut])
    out = kl.Activation('relu')(out)
    if use_se:
        out = se_block(out)
    if pool_size > 1:
        out = kl.MaxPooling1D(pool_size=pool_size, padding='same')(out)
    out = kl.Dropout(dropout_rate)(out)
    return out


def signal_encoder(input_shape, name='signal', l2_reg=1e-4):
    inp = kl.Input(shape=input_shape, name=f'{name}_input')

    # EVT-ADB: 自适应软阈值残差去噪前处理
    x = EVT_ADB_v2(input_shape[-1])(inp)

    x = kl.Conv1D(32, kernel_size=7, strides=2, padding='same',
                  kernel_regularizer=l2(l2_reg))(inp)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPooling1D(pool_size=3, strides=3, padding='same')(x)

    x = residual_block(x, 48, kernel_size=3, pool_size=3, dropout_rate=0.2, use_se=True, l2_reg=l2_reg)
    x = residual_block(x, 64, kernel_size=3, pool_size=3, dropout_rate=0.2, use_se=True, l2_reg=l2_reg)
    x = residual_block(x, 64, kernel_size=3, pool_size=2, dropout_rate=0.25, use_se=True, l2_reg=l2_reg)
    x = residual_block(x, 48, kernel_size=3, pool_size=1, dropout_rate=0.25, use_se=True, l2_reg=l2_reg)

    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Dropout(0.3)(x)

    model = keras.Model(inputs=inp, outputs=x, name=f'{name}_encoder')
    return model


def build_model(num_classes=2, signal_length=1125, stats_dim=23, l2_reg=1e-4):
    fhr_enc = signal_encoder((signal_length, 1), name='fhr', l2_reg=l2_reg)
    uc_enc = signal_encoder((signal_length, 1), name='uc', l2_reg=l2_reg)
    fm_enc = signal_encoder((signal_length, 1), name='fm', l2_reg=l2_reg)

    stats_input = kl.Input(shape=(stats_dim,), name='stats_input', dtype=np.float32)

    clinical = kl.Dense(24, activation='relu', kernel_regularizer=l2(l2_reg))(stats_input)
    clinical = kl.BatchNormalization()(clinical)
    clinical = kl.Dropout(0.2)(clinical)

    fused = kl.Concatenate()([fhr_enc.output, uc_enc.output, fm_enc.output, clinical])

    x = kl.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(fused)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.4)(x)
    x = kl.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.3)(x)

    output = kl.Dense(num_classes, activation='softmax', name='output',
                      kernel_regularizer=l2(l2_reg))(x)

    model = keras.Model(
        inputs=[fhr_enc.input, uc_enc.input, fm_enc.input, stats_input],
        outputs=output
    )
    return model


def build_model_simple(num_classes=2, signal_length=900, l2_reg=1e-4):
    fhr_enc = signal_encoder((signal_length, 1), name='fhr', l2_reg=l2_reg)
    uc_enc = signal_encoder((signal_length, 1), name='uc', l2_reg=l2_reg)
    fm_enc = signal_encoder((signal_length, 1), name='fm', l2_reg=l2_reg)

    fused = kl.Concatenate()([fhr_enc.output, uc_enc.output, fm_enc.output])

    x = kl.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(fused)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.4)(x)
    x = kl.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.3)(x)

    output = kl.Dense(num_classes, activation='softmax', name='output',
                      kernel_regularizer=l2(l2_reg))(x)

    model = keras.Model(
        inputs=[fhr_enc.input, uc_enc.input, fm_enc.input],
        outputs=output
    )
    return model


def build_feature_extractor(signal_length=1125, stats_dim=23, l2_reg=1e-4):
    fhr_enc = signal_encoder((signal_length, 1), name='fhr', l2_reg=l2_reg)
    uc_enc = signal_encoder((signal_length, 1), name='uc', l2_reg=l2_reg)
    fm_enc = signal_encoder((signal_length, 1), name='fm', l2_reg=l2_reg)

    stats_input = kl.Input(shape=(stats_dim,), name='stats_input', dtype=np.float32)

    clinical = kl.Dense(24, activation='relu', kernel_regularizer=l2(l2_reg))(stats_input)
    clinical = kl.BatchNormalization()(clinical)
    clinical = kl.Dropout(0.2)(clinical)

    fused = kl.Concatenate()([fhr_enc.output, uc_enc.output, fm_enc.output, clinical])

    x = kl.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(fused)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.4)(x)

    model = keras.Model(
        inputs=[fhr_enc.input, uc_enc.input, fm_enc.input, stats_input],
        outputs=x
    )
    return model
