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

from fairlearn.datasets import fetch_boston
import sklearn.metrics as skm
import fairlearn.metrics as fm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Assuming 1 is positive outcome, 1 is the protected class
PROTECTED_CLASS = 1
UNPROTECTED_CLASS = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0 

import numpy as np
import math

def get_and_preprocess_compas_data():
	"""Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
	
	Parameters
	----------
	params : Params
	Returns
	----------
	Pandas data frame X of processed data, np.ndarray y, and list of column names
	"""

	compas_X = pd.read_csv("dataset/compas-scores-two-years.csv", index_col=0)
	compas_X = compas_X.loc[(compas_X['days_b_screening_arrest'] <= 30) &
							  (compas_X['days_b_screening_arrest'] >= -30) &
							  (compas_X['is_recid'] != -1) &
							  (compas_X['c_charge_degree'] != "O") &
							  (compas_X['score_text'] != "NA")]

	compas_X['length_of_stay'] = (pd.to_datetime(compas_X['c_jail_out']) - pd.to_datetime(compas_X['c_jail_in'])).dt.days
	X = compas_X[['age','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

	# if person has high score give them the _negative_ model outcome
	y = np.array([NEGATIVE_OUTCOME if val == 1 else POSITIVE_OUTCOME for val in compas_X['two_year_recid']])

	sens = X.pop('race')

	# assign African-American as the protected class
	X = pd.get_dummies(X)
	sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
	X['race'] = sensitive_attr

	# make sure everything is lining up
	assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
	cols = [col for col in X]
	
	return X, y, [cols.index(val) for val in cols if val not in ['age', 'priors_count']]

