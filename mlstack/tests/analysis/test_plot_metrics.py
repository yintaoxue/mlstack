# coding=utf-8
"""
    test for plot metrics
    Created by xueyintao on 2018/8/5.
"""
import mlstack.pyplot as pyplot

y_test = [1,1,0,0,1,0,1,0,1,1,0,0,1,0]
y_pred = [1,1,0,0,1,0,1,0,1,0,1,0,1,0]
y_pred_prob = [0.8,0.6,0.3,0.2,0.6,0.1,0.9,0.4,0.7,0.3,0.7,0.2,0.8,0.3]


def test_plot_auc():
    pyplot.plot_auc(y_test, y_pred_prob)


def test_plot_ap():
    pyplot.plot_precision_recall_curve(y_test, y_pred)
    pyplot.plot_precision_recall_curve(y_test, y_pred_prob)


def test_plot_cm():
    pyplot.plot_confusion_matrix(y_test, y_pred)