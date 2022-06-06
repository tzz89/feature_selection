"""
This module contains the baseline feature importance calculation using native tree based model
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna
from xgboost import XGBRegressor


def unpack_search_dict(trial, search_dict):
    """ Generates the model parameters"""
    params = {}
    for key, value_dict in search_dict.items():
        if value_dict['type'] == 'categorical':
            params[key] = trial.suggest_categorical(name=key, choices=value_dict['choices'])
        elif value_dict['type'] == 'float':
            params[key] = trial.suggest_float(name=key, low=value_dict['low'], high=value_dict['high'])
        else:
            params[key] = trial.suggest_int(name=key, low=value_dict['low'], high=value_dict['high'])

    return params

def get_rf_regressor_feature_importance(features:np.ndarray, label, search_dict:dict, n_trials=1):
    """ 
    This function will train a random forest model using optuna 
    and return the best tuned model, the feature importance list and MSE and MAE

    search_dict schema 
    {
        parameter_name: {
            type:categorical, float, int
            low:
            high:
            choices: # if its categorical
        }
    }
    """
    
    def objective(trial):
        trial_params = unpack_search_dict(trial, search_dict)
        additional_params = {"n_jobs":-1, "random_state":24}
        full_params = {**trial_params, **additional_params}
        classifier = RandomForestRegressor(**full_params)
        mse = cross_val_score(classifier, features, label, scoring='neg_mean_squared_error')
        mse = [abs(value) for value in mse]
        return np.mean(np.sqrt(mse))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best_params")
    print(study.best_trial.params)

    #retraining with best params
    classifier = RandomForestRegressor(**study.best_trial.params)
    classifier.fit(features, label)
    return classifier.feature_importances_


def get_xgboost_regressor_feature_importance(features:np.ndarray, label, search_dict:dict, n_trials=1):
    """ 
    This function will train a xgboost model using optuna 
    and return the best tuned model, the feature importance list and MSE and MAE

    search_dict schema 
    {
        parameter_name: {
            type:categorical, float, int
            low:
            high:
            choices: # if its categorical
        }
    }
    """
    
    def objective(trial):
        trial_params = unpack_search_dict(trial, search_dict)
        additional_params = {"n_jobs":-1, "random_state":24}
        full_params = {**trial_params, **additional_params}
        classifier = XGBRegressor(**full_params)
        mse = cross_val_score(classifier, features, label, scoring='neg_mean_squared_error')
        mse = [abs(value) for value in mse]
        return np.mean(np.sqrt(mse))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("Best_params")
    print(study.best_trial.params)

    #retraining with best params
    classifier = XGBRegressor(**study.best_trial.params)
    classifier.fit(features, label)
    return classifier.feature_importances_