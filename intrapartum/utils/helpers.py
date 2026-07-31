import numpy as np
import tensorflow as tf
import time
import pickle
from datetime import datetime
from functools import partial
import keras
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
from data.dataset import dataset
from data.generators import MaskedDataGenerator, MaskedTestDataGenerator
from losses import binary_crossentropy_loss, sce_loss
from metrics.evaluation import f1, rec_pre, his_sum


def reset_random_seeds(seed_value=None):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def cosine_decay_with_warmup(global_step, learning_rate_base, total_steps,
                              warmup_learning_rate=0.0, warmup_steps=0,
                              hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(
        total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                  learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, learning_rate_base, total_steps, global_step_init=0,
                 warmup_learning_rate=0.0, warmup_steps=0, hold_base_rate_steps=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(
            global_step=self.global_step,
            learning_rate_base=self.learning_rate_base,
            total_steps=self.total_steps,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=self.warmup_steps,
            hold_base_rate_steps=self.hold_base_rate_steps,
        )
        keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning rate to %s.' % (self.global_step + 1, lr))


def reduce_lr_on_plateau(optimizer, val_loss, patience, factor, current_epoch, min_lr=1e-6):
    if hasattr(reduce_lr_on_plateau, 'best_val_loss'):
        if val_loss < reduce_lr_on_plateau.best_val_loss:
            reduce_lr_on_plateau.best_val_loss = val_loss
            reduce_lr_on_plateau.epochs_since_improvement = 0
        else:
            reduce_lr_on_plateau.epochs_since_improvement += 1
    else:
        reduce_lr_on_plateau.best_val_loss = val_loss
        reduce_lr_on_plateau.epochs_since_improvement = 0
    if reduce_lr_on_plateau.epochs_since_improvement >= patience:
        current_lr = optimizer.learning_rate
        new_lr = max(current_lr * factor, min_lr)
        optimizer.learning_rate.assign(new_lr)
        print(f"Epoch {current_epoch+1}: Learning rate reduced to {new_lr:.6f}")
        reduce_lr_on_plateau.epochs_since_improvement = 0
    else:
        new_lr = optimizer.learning_rate
    return new_lr


def evaluate_model(model, val_dataset):
    total_loss, total_accuracy, total_f1 = 0, 0, 0
    total_precision, total_recall = 0, 0
    for inputs, labels in val_dataset:
        predictions = model(inputs, training=False)
        loss = binary_crossentropy_loss(labels, predictions)
        total_loss += tf.reduce_mean(loss)
        predictions = tf.argmax(predictions, axis=-1)
        labels = tf.argmax(labels, axis=-1)
        labels = tf.cast(labels, tf.float32)
        predictions = tf.cast(predictions, tf.float32)
        total_accuracy += tf.reduce_mean(tf.keras.metrics.binary_accuracy(labels, predictions))
        total_f1 += f1.f1_scores(labels, predictions)
        total_precision += rec_pre.precision(labels, predictions)
        total_recall += rec_pre.recall(labels, predictions)
    num_batches = len(val_dataset)
    return (total_loss / num_batches, total_accuracy / num_batches,
            total_f1 / num_batches, total_precision / num_batches, total_recall / num_batches)


def train_model(model, train_dataset, epochs=1, batch_size=32, val_generator=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        with tqdm.tqdm(train_dataset, unit='batch', ncols=300) as pbar:
            for batch_idx, (inputs, labels) in enumerate(pbar):
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss = binary_crossentropy_loss(labels, predictions)
                    total_loss = loss
                    epoch_loss += total_loss
                gradients = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                pbar.set_postfix(loss=epoch_loss / (batch_idx + 1))
        print(f"Epoch {epoch + 1} loss: {epoch_loss / (batch_idx + 1):.4f}")
        if val_generator is not None:
            val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(model, val_generator)
            current_lr = reduce_lr_on_plateau(optimizer, val_loss, 2, 0.95, epoch)
            print(current_lr)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 Score: {val_f1:.4f}")
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")