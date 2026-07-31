import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from keras.engine import data_adapter
from losses.losses import NTXentLoss, sce_loss
from models.encoder import Encoder
from models.attention import AFF1D


class AdaptiveWeighting(tf.keras.Model):
    def __init__(self, initial_alpha=0.8, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.initial_alpha = initial_alpha
        self.alpha = tf.Variable(initial_alpha, trainable=False, dtype=tf.float32, name='alpha')
        self.prev_focal_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='prev_focal')
        self.prev_cont_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='prev_cont')

    def call(self, focal_loss, cont_loss):
        alpha_update = self.prev_focal_loss / (self.prev_focal_loss + self.prev_cont_loss + self.epsilon)
        self.prev_focal_loss.assign(focal_loss)
        self.prev_cont_loss.assign(cont_loss)
        return alpha_update


class TrainingModel(Model):
    def __init__(self, batch_size=32, random_remasking=False, augment=False,
                 dec_masking_ratio=0.2, mode='None', **kwargs):
        super(TrainingModel, self).__init__(**kwargs)
        self.nt_xent_criterion = NTXentLoss(batch_size, temperature=0.1,
                                             use_cosine_similarity=True, name="NTXentLoss")
        self.augment = augment
        self.adaptive_weight = AdaptiveWeighting()
        self.batch_size = batch_size
        self.batch_norm_196 = layers.BatchNormalization()
        self.batch_norm_128 = layers.BatchNormalization()
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.final_dense = layers.Dense(2, activation='softmax')
        self.random_remasking = random_remasking
        self.dec_masking_ratio = dec_masking_ratio
        self.mode = mode
        self.num_hidden = 128
        self.relu = layers.Activation('relu')
        self.encoder = Encoder()
        self.decoder = Dense(128)
        self.aff_1d1 = AFF1D(128, 4)
        self.aff_1d2 = AFF1D(128, 4)
        self.aff_1d3 = AFF1D(128, 4)

    def build(self, input_shapes):
        [batch_fhr1, batch_fhr2, batch_uc1, batch_uc2, batch_c] = input_shapes
        self.input_layer_1 = layers.Input(shape=batch_fhr1)
        self.input_layer_2 = layers.Input(shape=batch_fhr2)
        self.input_layer_3 = layers.Input(shape=batch_uc1)
        self.input_layer_4 = layers.Input(shape=batch_uc2)
        self.input_layer_5 = layers.Input(shape=batch_c)

    def call(self, inputs, training=False, return_embedding=False):
        if self.augment:
            i1, j1, i2, j2, ci, cj = inputs
            i1 = tf.expand_dims(tf.expand_dims(i1, axis=1), axis=3)
            ci = tf.expand_dims(ci, axis=1)
            j1 = tf.expand_dims(tf.expand_dims(j1, axis=1), axis=3)
            cj = tf.expand_dims(cj, axis=1)
            i2 = tf.expand_dims(tf.expand_dims(i2, axis=1), axis=3)
            j2 = tf.expand_dims(tf.expand_dims(j2, axis=1), axis=3)
            i1, i2, ci = self.encoder(i1, i2, ci)
            j1, j2, cj = self.encoder(j1, j2, cj)
            i1 = self.batch_norm_196(i1)
            j1 = self.batch_norm_196(j1)
            i1 = self.relu(i1)
            i1 = self.decoder(i1)
            j1 = self.relu(j1)
            j1 = self.decoder(j1)
            if self.mode == 'None':
                outputs = tf.concat([i1, i2, cj], axis=2)
            elif self.mode == 'AFF':
                if self.random_remasking:
                    masked_data_i1, masked_data_j1, masked_data_i2, masked_data_j2, \
                    masked_clinic_data_i, masked_clinic_data_j = self.dec_remask(
                        i1, j1, i2, j2, ci, cj,
                        mask_feat_ratio=self.dec_masking_ratio,
                        pos_mask_ratio=self.dec_masking_ratio,
                        neg_mask_ratio=self.dec_masking_ratio,
                    )
                else:
                    masked_data_i1, masked_data_j1, masked_data_i2, masked_data_j2, \
                    masked_clinic_data_i, masked_clinic_data_j = i1, j1, i2, j2, ci, cj
                net1_Res = tf.expand_dims(self.global_avg_pool(masked_data_i1), axis=1)
                net2_Res = tf.expand_dims(self.global_avg_pool(masked_data_i2), axis=1)
                outputs1 = self.aff_1d1(net2_Res, masked_clinic_data_i)
                outputs2 = self.aff_1d2(net1_Res, masked_clinic_data_i)
                outputs3 = self.aff_1d3(net1_Res, net2_Res)
                outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
                outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
                outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)
                outputs_1 = tf.concat([outputs1, outputs2, outputs3], 2)
                outputs_1 = self.global_avg_pool(outputs_1)
                outputs_1 = self.final_dense(outputs_1)
                net1_Res = tf.expand_dims(self.global_avg_pool(masked_data_j1), axis=1)
                net2_Res = tf.expand_dims(self.global_avg_pool(masked_data_j2), axis=1)
                outputs1 = self.aff_1d1(net2_Res, masked_clinic_data_j)
                outputs2 = self.aff_1d2(net1_Res, masked_clinic_data_j)
                outputs3 = self.aff_1d3(net1_Res, net2_Res)
                outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
                outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
                outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)
                outputs_2 = tf.concat([outputs1, outputs2, outputs3], 2)
                outputs_2 = self.global_avg_pool(outputs_2)
                outputs_2 = self.final_dense(outputs_2)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            i1 = self.global_avg_pool(i1)
            j1 = self.global_avg_pool(j1)
            i2 = self.global_avg_pool(i2)
            j2 = self.global_avg_pool(j2)
            ci = self.global_avg_pool(ci)
            cj = self.global_avg_pool(cj)
            return outputs_1, outputs_2
        else:
            if len(inputs) == 5:
                i1, j1, i2, j2, c = inputs
                i1, i2, ci1 = self.encoder(i1, i2, c, training)
                j1, j2, ci2 = self.encoder(j1, j2, c, training)
                i1 = self.batch_norm_196(i1)
                i1 = self.relu(i1)
                i1 = self.decoder(i1)
                j1 = self.batch_norm_196(j1)
                j1 = self.relu(j1)
                j1 = self.decoder(j1)
                if self.mode == 'None':
                    i1 = tf.expand_dims(self.global_avg_pool(i1), axis=1)
                    i2 = tf.expand_dims(self.global_avg_pool(i2), axis=1)
                    outputs = tf.concat([i1, i2, ci1], axis=2)
                elif self.mode == 'AFF':
                    if self.random_remasking:
                        num_hidden = 128
                        num_samples = int(375 * 0.1)
                        num_features = int(num_hidden * 0.1)
                        batch_size = tf.shape(i1)[0]

                        def mask_single_sample(idx):
                            selected_rows = tf.random.shuffle(tf.range(375))[:num_samples]
                            indices = []
                            for row in selected_rows.numpy():
                                selected_features = tf.random.shuffle(tf.range(num_hidden))[:num_features]
                                for col in selected_features.numpy():
                                    indices.append([row, col])
                            indices_tensor = tf.convert_to_tensor(indices, dtype=tf.int32)
                            mask_values = tf.zeros([num_samples * num_features], dtype=tf.float32)
                            masked = tf.tensor_scatter_nd_update(i1[idx], indices_tensor, mask_values)
                            return masked

                        masked_data_i1 = tf.vectorized_map(mask_single_sample, tf.range(batch_size))
                        i1 = masked_data_i1
                        i2 = masked_data_i1

                    outputs1 = self.aff_1d1(i1, ci1)
                    outputs2 = self.aff_1d2(i2, ci1)
                    outputs3 = self.aff_1d3(i1, i2)
                    outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
                    outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
                    outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)
                    outputs_1 = tf.concat([outputs1, outputs2, outputs3], 2)
                    outputs1 = self.aff_1d1(j1, ci2)
                    outputs2 = self.aff_1d2(j2, ci2)
                    outputs3 = self.aff_1d3(j1, j2)
                    outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
                    outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
                    outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)
                    outputs_2 = tf.concat([outputs1, outputs2, outputs3], 2)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                outputs_1 = self.global_avg_pool(outputs_1)
                outputs_1 = self.final_dense(outputs_1)
                if return_embedding:
                    return outputs_1
                return outputs_1
            else:
                i1, i2, c = inputs
                i1, i2, ci = self.encoder(i1, i2, c, training)
                i1 = self.batch_norm_196(i1)
                i1 = self.relu(i1)
                i1 = self.decoder(i1)
                if self.mode == 'None':
                    i1 = tf.expand_dims(self.global_avg_pool(i1), axis=1)
                    i2 = tf.expand_dims(self.global_avg_pool(i2), axis=1)
                    outputs = tf.concat([i1, i2, ci], axis=2)
                elif self.mode == 'AFF':
                    outputs1 = self.aff_1d1(i1, ci)
                    outputs2 = self.aff_1d2(i2, ci)
                    outputs3 = self.aff_1d3(i1, i2)
                    outputs1 = tf.expand_dims(self.global_avg_pool(outputs1), axis=1)
                    outputs2 = tf.expand_dims(self.global_avg_pool(outputs2), axis=1)
                    outputs3 = tf.expand_dims(self.global_avg_pool(outputs3), axis=1)
                    outputs = tf.concat([outputs1, outputs2, outputs3], 2)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                outputs = self.global_avg_pool(outputs)
                if return_embedding:
                    return outputs
                outputs = self.final_dense(outputs)
                return outputs

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight,
                                     regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        del x
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del x
        return self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)