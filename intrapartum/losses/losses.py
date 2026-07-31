import tensorflow as tf
import numpy as np
from keras import backend as K


class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size, temperature, use_cosine_similarity=True, name="NTXentLoss"):
        super(NTXentLoss, self).__init__(name=name)
        self.batch_size = batch_size
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.mask_samples_from_same_repr = self._get_correlated_mask()
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM,
        )

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = diag + l1 + l2
        mask = 1 - mask
        return tf.convert_to_tensor(mask, dtype=tf.bool)

    def _cosine_similarity(self, x, y):
        x_normalized = tf.nn.l2_normalize(x, axis=-1)
        y_normalized = tf.nn.l2_normalize(y, axis=-1)
        return tf.linalg.matmul(x_normalized, y_normalized, transpose_b=True)

    def _dot_similarity(self, x, y):
        return tf.linalg.matmul(x, y, transpose_b=True)

    def call(self, zis, zjs):
        pad_rows_zis = 32 - tf.shape(zis)[0]
        pad_rows_zjs = 32 - tf.shape(zjs)[0]
        pad_zis = tf.zeros([pad_rows_zis, 128], dtype=zis.dtype)
        pad_zjs = tf.zeros([pad_rows_zjs, 128], dtype=zjs.dtype)
        zis_padded = tf.concat([zis, pad_zis], axis=0)
        zjs_padded = tf.concat([zjs, pad_zjs], axis=0)
        representations = tf.concat([zjs_padded, zis_padded], axis=0)
        if self.use_cosine_similarity:
            similarity_matrix = self._cosine_similarity(representations, representations)
        else:
            similarity_matrix = self._dot_similarity(representations, representations)
        l_pos = tf.linalg.diag_part(similarity_matrix, k=self.batch_size)
        r_pos = tf.linalg.diag_part(similarity_matrix, k=-self.batch_size)
        positives = tf.concat([l_pos, r_pos], axis=0)
        positives = tf.expand_dims(positives, axis=1)
        negatives = tf.boolean_mask(similarity_matrix, self.mask_samples_from_same_repr)
        negatives = tf.reshape(negatives, [2 * self.batch_size, -1])
        logits = tf.concat([positives, negatives], axis=1)
        logits /= self.temperature
        labels = tf.zeros(2 * self.batch_size, dtype=tf.int32)
        loss = self.criterion(labels, logits)
        return loss / (2 * tf.cast(self.batch_size, tf.float32))


def sce_loss(x, y, alpha=3):
    x = tf.math.l2_normalize(x, axis=-1)
    y = tf.math.l2_normalize(y, axis=-1)
    loss = (1 - tf.reduce_sum(x * y, axis=-1)) ** alpha
    loss = tf.reduce_mean(loss)
    return loss


def binary_crossentropy_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    false_negative_penalty = 1.5
    fn_penalty = false_negative_penalty * tf.reduce_mean(
        tf.multiply(y_true, (1 - y_pred)),
    )
    total_loss = bce_loss + fn_penalty
    return total_loss


def dice_loss(y_true, y_pred):
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred)
    return 1.0 - numerator / (denominator + K.epsilon())


def bce_dice_loss(y_true, y_pred):
    bce_loss = binary_crossentropy_loss(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return bce_loss + dice_loss_val


def focal_loss(gamma=2.0, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        fl = alpha * y_true * tf.math.pow(1 - y_pred, gamma) * ce \
           + (1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma) * ce
        return tf.reduce_mean(fl)
    return focal_loss_fixed