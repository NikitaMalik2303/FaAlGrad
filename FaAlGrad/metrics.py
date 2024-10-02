import torch
from torch import autograd as ag

import numpy as np
import pandas as pd
import math
import os
import pickle
import shutil
import zipfile
from urllib.request import urlretrieve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_ratio, equalized_odds_difference, demographic_parity_ratio, true_positive_rate


# Assuming 1 is positive outcome, 1 is the protected class
PROTECTED_CLASS = 1
UNPROTECTED_CLASS = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0 

import numpy as np
import math

def equal_odds_diff(Y, Y_hat, A):
    Y = Y.numpy()
    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()
    
    A = np.where(A == 1, 'P', 'U')
    
    eo_diff = equalized_odds_difference(y_true=Y, y_pred=Y_hat, sensitive_features=A)
 
    return eo_diff


    
def demographic_parity_diff(Y, Y_hat, A):
    
    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()
    A = np.where(A == 1, 'P', 'U')
    
    # Compute the disparate impact difference
    dp_diff = demographic_parity_difference(y_true=Y, y_pred=Y_hat, sensitive_features=A)
    
    return dp_diff

def equal_odds_ratio(Y, Y_hat, A):
    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()
    A = np.where(A == 1, 'P', 'U')

    eo_ratio = equalized_odds_ratio(y_true=Y, y_pred=Y_hat, sensitive_features=A)

    return eo_ratio


def demographic_parity_ratio_(Y, Y_hat, A):
    
    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()
    A = np.where(A == 1, 'P', 'U')
    
    # Compute the disparate impact ratio
    dp_ratio = demographic_parity_ratio(y_true=Y, y_pred=Y_hat, sensitive_features=A)
    
    return dp_ratio

def true_positive_rate(Y, Y_hat, A):
    Y = Y.squeeze()
    Y_hat = Y_hat.squeeze()
    A = np.where(A == 1, 'P', 'U')
    tp_rate = true_positive_rate(y_true=Y, y_pred=Y_hat, sensitive_features=A)
    return tp_rate



def totorch(x, device,grad=False):
	return ag.Variable(torch.Tensor(x),requires_grad=grad).to(device)
