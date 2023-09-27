from sklearn import metrics
import numpy as np
import torch

def accuracy(pred, y_true):
    pred = np.argmax(pred, axis= 1)

    return metrics.accuracy_score(pred, y_true)

def balanced_accuracy(pred, y_true):
    pred = np.argmax(pred, axis= 1)

    return metrics.balanced_accuracy_score(pred, y_true)