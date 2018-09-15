# coding=utf-8
"""
    classifcation metrics
    Created by xueyintao on 2018/9/6.
"""
from sklearn import metrics
import numpy as np
import pandas as pd

def get_precision_recall_by_threshold(threshold, y_test, y_score):
    """
    get precision and recall at given threshold
    :param threshold:
    :param y_test:
    :param y_score:
    :return:
        p: precision
        r: recall
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

    pdf = pd.DataFrame(np.array(precision), columns=['precision'])
    rdf = pd.DataFrame(np.array(recall), columns=['recall'])
    tdf = pd.DataFrame(np.array(thresholds), columns=['thresholds'])

    prt = pd.concat([pdf, rdf, tdf], axis=1)

    prt1 = prt.loc[prt['thresholds'] >= threshold, :].iloc[0, :]
    p = prt1[0]
    r = prt1[1]
    t = prt1[2]

    return p, r

def get_precision_by_recall(recall, y_test, y_score):
    """
    get precision and threshold at given recall
    :param recall
    :param y_test:
    :param y_score:
    :return:
        p: precision
        t: threshold
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

    pdf = pd.DataFrame(np.array(precision), columns=['precision'])
    rdf = pd.DataFrame(np.array(recall), columns=['recall'])
    tdf = pd.DataFrame(np.array(thresholds), columns=['thresholds'])

    prt = pd.concat([pdf, rdf, tdf], axis=1)

    prt1 = prt.loc[prt['recall'] >= recall, :].iloc[-1, :]
    p = prt1[0]
    r = prt1[1]
    t = prt1[2]

    return p, t


def get_recall_by_precision(precision, y_test, y_score):
    """
    get recall and threshold at given precision
    :param precision:
    :param y_test:
    :param y_score:
    :return:
        r: recall
        t: threshold
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_score)

    pdf = pd.DataFrame(np.array(precision), columns=['precision'])
    rdf = pd.DataFrame(np.array(recall), columns=['recall'])
    tdf = pd.DataFrame(np.array(thresholds), columns=['thresholds'])

    prt = pd.concat([pdf, rdf, tdf], axis=1)

    prt1 = prt.loc[prt['precision'] >= precision, :].iloc[0, :]
    p = prt1[0]
    r = prt1[1]
    t = prt1[2]

    return r, t
