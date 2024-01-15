# -*- coding: utf-8 -*-
"""
Define common metrics.
"""

# Third party import
import logging
import torch
import numpy as np
import scipy, re
from logs.utils import get_pickle_obj
from typing import List, Dict, Sequence
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, accuracy_score, average_precision_score
from scipy.special import expit

# Global parameters
logger = logging.getLogger()

def get_confusion_matrix(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.argmax(axis=1)
    return confusion_matrix(y_true=y_true, y_pred=y_pred)

def accuracy(y_pred, y_true, threshold=0):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > threshold)
    elif len(y_pred.shape) == 2:
        y_pred = y_pred.argmax(axis=1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)

def average_precision(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return average_precision_score(y_true=y_true, y_score=y_pred)

def roc_auc(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        return roc_auc_score(y_true=y_true, y_score=y_pred[:,1])
    elif len(y_pred.shape) < 2:
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    else:
        raise ValueError('Invalid shape for y_pred: {}'.format(y_pred.shape))

def balanced_accuracy(y_pred, y_true, threshold=0):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > threshold)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.argmax(axis=1) # get the indices of the maximum
    return balanced_accuracy_score(y_true=y_true, y_pred=y_pred)

# True Positive Rate = TP/P (also called Recall)
def sensitivity(y_pred, y_true, positive_label=1):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.argmax(axis=1)
    TP = ((y_pred == y_true) & (y_true == positive_label)).sum()
    P = (y_true == positive_label).sum()
    if P == 0:
        return 0.0
    return float(TP/P)

# True Negative Rate = TN/N
def specificity(y_pred, y_true, negative_label=0):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.argmax(axis=1)
    TN = ((y_pred == y_true) & (y_true == negative_label)).sum()
    N = (y_true == negative_label).sum()
    if N == 0:
        return 0.0
    return float(TN/N)


def RMSE(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
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
