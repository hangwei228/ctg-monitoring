import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef)
import matplotlib.pyplot as plt


class rec_pre:
    @staticmethod
    def recall(y_true, y_pred):
        TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        FN = np.sum(np.round(np.clip(y_true, 0, 1)))
        return TP / (FN + 1e-8)

    @staticmethod
    def precision(y_true, y_pred):
        TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        FP = np.sum(np.round(np.clip(y_pred, 0, 1)))
        return TP / (FP + 1e-8)


class f1(rec_pre):
    @staticmethod
    def f1_scores(y_true, y_pred):
        pre = rec_pre.precision(y_true, y_pred)
        rec = rec_pre.recall(y_true, y_pred)
        return 2 * ((pre * rec) / (pre + rec + 1e-8))


class his_sum:
    @staticmethod
    def score_sum(data_y_test, pred, pred1):
        print('Confusion Matrix:', np.array(confusion_matrix(data_y_test, pred)).T, sep='\n')
        print('\nThe Classification Report:', classification_report(data_y_test, pred), sep='\n')
        print('Kappa Value:', cohen_kappa_score(data_y_test, pred))
        print('Matthews correcoef:', matthews_corrcoef(data_y_test, pred))
        auc_value = roc_auc_score(data_y_test, pred1)
        print('\nAuc面积为：', auc_value)
        fpr, tpr, thresholds = roc_curve(data_y_test, pred1)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', linewidth=2, label='ROC,curve(area = %0.4f)' % auc_value)
        plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Example')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        return cohen_kappa_score(data_y_test, pred)

    @staticmethod
    def paint_history(history, filename):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['f1_scores'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['val_f1_scores'])
        plt.title('Model train VS validation')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'train_f1', 'val_loss', 'val_f1'], loc=2,
                    bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(filename + '.jpg')
        plt.show()