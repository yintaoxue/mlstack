# coding=utf-8
"""
    
    Created by xueyintao on 2018/9/6.
"""
import mlstack.metrics as metrics

y_test = [1,1,0,0,1,0,1,0,1,1,0,0,1,0]
y_pred = [1,1,0,0,1,0,1,0,1,0,1,0,1,0]
y_pred_prob = [0.8,0.6,0.3,0.2,0.6,0.1,0.9,0.4,0.7,0.3,0.7,0.2,0.8,0.3]


def test_get_precision_recall_by_threshold():
    metrics.get_recall_by_precision()
    print()

