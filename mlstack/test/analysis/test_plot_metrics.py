# coding=utf-8
"""
    test for plot metrics
    Created by xueyintao on 2018/8/5.
"""
from analysis import plot_metrics

y_test = [1,1,0,0,1,0,1,0,1,1,0,0,1,0]
y_pred = [1,1,0,0,1,0,1,0,1,0,1,0,1,0]
y_pred_prob = [0.8,0.6,0.3,0.2,0.6,0.1,0.9,0.4,0.7,0.3,0.7,0.2,0.8,0.3]


def test_plot_auc():
    plot_metrics.plot_auc(y_test, y_pred_prob)


def test_plot_ap():
    plot_metrics.plot_precision_recall_curve(y_test, y_pred)
    plot_metrics.plot_precision_recall_curve(y_test, y_pred_prob)


def test_plot_cm():
    plot_metrics.plot_confusion_matrix(y_test, y_pred)