### RandomgridSearchCV to tune lgb for all three subset 
### 3 data subset x 3 quantiles = 9 total models fitted

N_HYPER = 150
SEED = 1337
N_JOBS = 4 
N_ROWS= None
HYPER_GRID = {
    'num_leaves': np.arange(5, 300, 3),
    'min_child_samples': np.arange(100, 3000, 100),
    'min_child_weight': np.logspace(-5, 4, 10),
    'subsample': np.linspace(0.2, 1, 15),
    'colsample_bytree': np.linspace(0.45, 1, 15),
    'reg_lambda': np.logspace(-1, 3, 15)
}

import sys
import importlib
import re
import numpy as np
import pandas as pd
from wifipricing.modeling_prepartions import get_lgb_data
from wifipricing.model_tuning import lgb_random_search

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
from datetime import datetime

data_paths = {
    'datacap':'data/summarized_data/summarized_datacap.csv',
    'timecap':'data/summarized_data/summarized_timecap.csv',
    'fulldata':'data/summarized_data/summarized_all.csv'
}

# Outerloop: 3 data subset
for subset, path in data_paths.items():
    df = pd.read_csv(path, nrows=N_ROWS)

    X, y = get_lgb_data(df, subset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('X train\n', X_train.dtypes)
    print('\n\nX test\n', X_test.dtypes)

    quantiles = {'lower': 0.05, 'median':0.5, 'upper':0.95}

    start = time.time()
    # innerloop: 3 quantile regression alphas 
    for quantile, alp in quantiles.items():
        print(f'\n\n Fitting {k} quantile: {X_train.shape}\n')
        # Make sure that alpha is reset after each run
        lgb_reg = lgb.LGBMRegressor(random_state=SEED, n_jobs=4, objective='quantile')

        # Validation
        rcv = lgb_random_search(X_train, X_test, y_train, y_test, lgb_reg, alp, HYPER_GRID, SEED, cv=5)
        y_validate = rcv.predict(X_test)

        # Refitting model with best param/full dataset 
        lgb_reg = lgb.LGBMRegressor(random_state=SEED, n_jobs=4, objective='quantile')
        lgb_reg.set_params(alpha = alp)
        lgb_reg.set_params(**rcv.best_params_)

        print(f'\n\n Refitting on full dataset with the following parameters:')
        print(lgb_reg.get_params())

        refit_params={'categorical_feature': 'auto'}
        model = lgb_reg.fit(X, y, **refit_params)

        quantiles[quantile] = {'model': model,
                               'y_test': y_test,
                               'y_predict': y_validate,
                               'training_data_size': X_train.shape,
                               'cv_results': rcv.cv_results_}

    end = time.time()
    print(f'Fitting by random grid search for {subset} took {round((end - start) / 60)} minutes')

    # saving results
    curtime = datetime.now().strftime("%Y%m%d_%H_%M")  
    prefix = f'models/lgb_randgrid_{subset}'

    for quantile, item in quantiles.items():
        file = prefix + f'_quantile_{quantile}_' + curtime+ ".joblib"
        joblib.dump(item,  file)