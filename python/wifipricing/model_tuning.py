from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt 
import time



def lgb_random_search(
        X_train, X_test, y_train, y_test, 
        regressor, alp, hyper_grid, n_hyper, seed, 
        categorical_features, metrics, cv=5):
    """sklearn RandomizedSearchCV wrapper (uses Label Encoded data)"""
    # This passes additional args that are not in LGBMModel args
    fit_params={
        'eval_set' : [(X_test, y_test)],
        'eval_names': ['validation set'],
        'eval_metric': metrics,
        'early_stopping_rounds':30,
        'feature_name': list(X_test.columns),
        'categorical_feature': categorical_features,
        'verbose':[10]
    }

    regressor.set_params(alpha = alp)

    rcv = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=hyper_grid,
        n_iter=n_hyper,
        cv=cv,
        random_state=seed,
        verbose=2
    )

    rcv.fit(X_train, y_train, **fit_params)

    return rcv


def lgb_random_search_onehot(
        X_train, X_test, y_train, y_test, 
        regressor, alp, hyper_grid, n_hyper, seed, 
        categorical_features, metrics, cv=5):
    """sklearn RandomizedSearchCV wrapper (uses OneHot Encoded data)"""
    # This passes additional args that are not in LGBMModel args
    fit_params={
        'eval_set' : [(X_test, y_test)],
        'eval_names': ['validation set'],
        'eval_metric': metrics,
        'early_stopping_rounds':30,
        'feature_name': list(X_test.columns),
        # 'categorical_feature': categorical_features,
        'verbose':[10]
    }

    regressor.set_params(alpha = alp)

    rcv = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=hyper_grid,
        n_iter=n_hyper,
        cv=cv,
        random_state=seed,
        verbose=2
    )

    rcv.fit(X_train, y_train, **fit_params)

    return rcv

def rmse(y_true, y_pred):
    """returns (eval_name, eval_result, is_bigger_better)"""
    result = sqrt(mean_squared_error(y_true, y_pred))

    return ('RMSE', result, False)
