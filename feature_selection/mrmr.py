"""
This module holds the code for the calculation of MRMR
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from xgboost import XGBRegressor
import numpy as np

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def generate_xgboost_score(features, label, feature_list):
    score_list = []
    for i in range(1, len(feature_list)):
        regressor = XGBRegressor(n_jobs=-1)
        mse = cross_val_score(regressor, features[:,:i], label, scoring='neg_mean_squared_error')
        mse = [abs(value) for value in mse]
        score_list.append(np.mean(np.sqrt(mse)))
    return score_list

def generate_rf_score(features, label, feature_list):
    score_list = []
    for i in range(1, len(feature_list)):
        regressor = RandomForestRegressor(n_jobs=-1)
        mse = cross_val_score(regressor, features[:,:i], label, scoring='neg_mean_squared_error')
        mse = [abs(value) for value in mse]
        score_list.append(np.mean(np.sqrt(mse)))
    return score_list