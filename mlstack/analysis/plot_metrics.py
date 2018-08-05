# coding=utf-8
"""
    Plot model metrics
    Created by xueyintao on 2018/8/5.
"""
import sklearn.metrics as metrics
import matplotlib.pyplot as plot
import numpy as np
import itertools


def plot_auc(y_test, y_score):
    """
    plot auc
    :param y_test:
    :param y_score: the predicted prob
    :return:
    """
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plot.title('AUC')
    plot.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plot.legend(loc = 'lower right')
    plot.plot([0, 1], [0, 1],'r--')
    plot.xlim([0, 1])
    plot.ylim([0, 1])
    plot.ylabel('True Positive Rate')
    plot.xlabel('False Positive Rate')
    plot.show()


def plot_precision_recall_curve(y_test, y_score):
    """
    plot precision recall curve
    :param y_test:
    :param y_score: the predicted prob
    :return:
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)
    average_precision = metrics.average_precision_score(y_test, y_score)

    plot.step(recall, precision, color='b', alpha=0.2,
              where='post')
    plot.fill_between(recall, precision, step='post', alpha=0.2,
                      color='b')

    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.ylim([0.0, 1.05])
    plot.xlim([0.0, 1.0])
    plot.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plot.show()


def plot_confusion_matrix(y_true, y_pred, classes=['0','1'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):
    """
    plot confusion matrix

    :param y_true:
    :param y_pred: the predicted label
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    cm = metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.show()