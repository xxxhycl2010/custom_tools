# coding:utf-8

'''

this script aims to evalute model and draw p-r curve and roc curve .
Author: chenglong Yi
date: 2019-02-15

'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc 
from scipy import interp
from itertools import cycle

from draw_curve_helper import DrawCurveHelper


helper = DrawCurveHelper()

def test_1():
    filepath = "../data/single_cls_res.txt"
    
    helper.draw_pr_curve(filepath)
    helper.draw_roc_curve(filepath)

def test_2():
    filepath = "../data/multi_cls_res.txt"
    helper.draw_pr_curve_multi(filepath, True)
    helper.draw_roc_curve_multi(filepath, True)





if __name__ == "__main__":
    test_1()
    # test_2()

 




