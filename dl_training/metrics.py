# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define common metrics.
"""

# Third party import
import logging
import torch
import numpy as np
import sklearn.metrics as sk_metrics
import scipy, re
from logs.utils import get_pickle_obj
from typing import List, Dict, Sequence
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from scipy.special import expit

# Global parameters
logger = logging.getLogger()

def get_confusion_matrix(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return confusion_matrix(y, y_pred.detach().cpu().numpy())

def accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    y_pred = y_pred.data.max(dim=1)[1]
    return sk_metrics.accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def average_precision_score(y_pred, y):
    return sk_metrics.average_precision_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def roc_auc(y_pred, y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        return roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())
    elif len(y_pred.shape) < 2:
        return roc_auc_score(y, y_pred.detach().cpu().numpy())
    else:
        raise ValueError('Invalid shape for y_pred: {}'.format(y_pred.shape))

def balanced_accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.data.max(dim=1)[1] # get the indices of the maximum
    return balanced_accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

# True Positive Rate = TP/P (also called Recall)
def sensitivity(y_pred, y, positive_label=1):
    y_pred = y_pred.data.max(dim=1)[1]
    TP = (y_pred.eq(y) & y.eq(positive_label)).sum().cpu().numpy()
    P = y.eq(positive_label).sum().cpu().numpy()
    if P == 0:
        return 0.0
    return float(TP/P)

# True Negative Rate = TN/N
def specificity(y_pred, y, negative_label=0):
    y_pred = y_pred.data.max(dim=1)[1]
    TN = (y_pred.eq(y) & y.eq(negative_label)).sum().cpu().numpy()
    N = y.eq(negative_label).sum().cpu().numpy()
    if N == 0:
        return 0.0
    return float(TN/N)


def RMSE(y_pred, y):
    rmse = torch.sqrt(torch.mean((y_pred - y)**2)).detach().cpu().numpy()
    return float(rmse)


METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "average_precision_score": average_precision_score,
    "RMSE": RMSE,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "confusion_matrix": get_confusion_matrix,
    "roc_auc": roc_auc,
    # cf. scikit doc: " The binary case expects a shape (n_samples,), and the scores
    # must be the scores of the class with the greater label."
}
