# -*- coding: utf-8 -*-
"""
DAST-MMNet Training Entry Point
"""
import os
import time
import pickle
from datetime import datetime
import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm

from config import (BATCH_SIZE, EPOCH, MODE, MODEL_NAME, RESULT_DIR,
                    CUDA_VISIBLE_DEVICES, INPUT_SHAPE_1, INPUT_SHAPE_2, OPTIMIZER, LR)
from models.training_model import TrainingModel
from data.dataset import dataset
from data.generators import MaskedDataGenerator, MaskedTestDataGenerator
from losses import binary_crossentropy_loss
from metrics.evaluation import f1, rec_pre, his_sum

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
tf.compat.v1.reset_default_graph()

dirs = RESULT_DIR
if not os.path.exists(dirs):
    os.makedirs(dirs)


def reduce_lr_on_plateau(optimizer, val_loss, patience, factor, current_epoch, min_lr=1e-6):
    if not hasattr(reduce_lr_on_plateau, 'best_val_loss'):
        reduce_lr_on_plateau.best_val_loss = val_loss
        reduce_lr_on_plateau.epochs_since_improvement = 0
    elif val_loss < reduce_lr_on_plateau.best_val_loss:
        reduce_lr_on_plateau.best_val_loss = val_loss
        reduce_lr_on_plateau.epochs_since_improvement = 0
    else:
        reduce_lr_on_plateau.epochs_since_improvement += 1
    if reduce_lr_on_plateau.epochs_since_improvement >= patience:
        current_lr = optimizer.learning_rate
        new_lr = max(current_lr * factor, min_lr)
        optimizer.learning_rate.assign(new_lr)
        print(f"Epoch {current_epoch+1}: Learning rate reduced to {new_lr:.6f}")
        reduce_lr_on_plateau.epochs_since_improvement = 0
    return optimizer.learning_rate


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
        total_accuracy += tf.reduce_mean(
            tf.keras.metrics.binary_accuracy(labels, predictions),
        )
        total_f1 += f1.f1_scores(labels, predictions)
        total_precision += rec_pre.precision(labels, predictions)
        total_recall += rec_pre.recall(labels, predictions)
    num_batches = len(val_dataset)
    return (total_loss / num_batches, total_accuracy / num_batches,
            total_f1 / num_batches, total_precision / num_batches, total_recall / num_batches)


class training:
    def fit_models(self, dir, model_name, epoch=60, mode='None',
                   optimizer='Adam', batch_size=128,
                   input_shape_1=(2, 6000, 1), input_shape_2=(1, 14), times=1):
        evaluator = his_sum()
        for i in range(2, 3):
            tf.random.set_seed(i)
            t1 = time.localtime(time.time())
            filename = (dir + str(t1.tm_year) + str(t1.tm_mon) + str(t1.tm_mday) +
                        str(t1.tm_hour) + str(t1.tm_min))
            t = time.time()
            logdir = "logs" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filename + model_name + "_best.ckpt",
                    save_best_only=True, monitor="val_loss",
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.92, patience=2, min_lr=0, verbose=True,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=50, verbose=1,
                ),
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.TensorBoard(log_dir=logdir),
            ]
            opt = keras.optimizers.Adam(learning_rate=LR)
            model = TrainingModel(
                batch_size=batch_size, mode=mode,
                augment=False, random_remasking=False,
            )
            print('\nBuilding success!\n\nLoading data...\n')
            data_x_train, data_c_train, data_y_train = dataset.Load_train()
            data_x_val, data_c_val, data_y_val = dataset.Load_val()
            print('Loading Success!\n\nTraining model...\n')
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                metrics=[
                    'binary_accuracy', f1.f1_scores, rec_pre.recall, rec_pre.precision,
                ],
            )
            train_generator = MaskedDataGenerator(
                data_x=data_x_train, data_c=data_c_train, data_y=data_y_train,
                batch_size=batch_size,
            )
            val_generator = MaskedTestDataGenerator(
                data_x=data_x_val, data_c=data_c_val, data_y=data_y_val,
                batch_size=batch_size,
            )
            history = model.fit(
                x=train_generator, epochs=epoch, validation_data=val_generator,
                shuffle=True, callbacks=callbacks, verbose=1,
            )
            print('Training sucess!\nSaving model...\n')
            print(filename)
            with open(filename + model_name + '_history.txt', 'wb') as f:
                pickle.dump(history.history, f)
            model.save(filename + model_name + '_model.ckpt')
            print("Saving sucess!\n")
            print('1 epoch spend：' + str((time.time() - t) / epoch) + ' s')
            his_sum.paint_history(history, filename)
            evaluator = his_sum()
            evaluator.Scoring(model=model, batch_size=batch_size)
        print("Best score after", times, "training cycles:", evaluator.get_best_score())


if __name__ == '__main__':
    trainer = training()
    dense_output_data = trainer.fit_models(
        dir=dirs, mode=MODE, model_name=MODEL_NAME,
        batch_size=BATCH_SIZE, epoch=EPOCH, optimizer=OPTIMIZER,
        input_shape_1=INPUT_SHAPE_1, input_shape_2=INPUT_SHAPE_2,
    )
    print('finish')