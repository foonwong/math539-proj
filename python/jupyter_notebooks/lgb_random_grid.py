#%%
N_HYPER = 400 
SEED = 1337
DEVICE = 'gpu'
N_JOBS = 4 
N_ROWS= None
# N_ROWS= 10000


#%%
import sys
import importlib
import re
import numpy as np
import pandas as pd

# Notebook options
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%%
df_data = pd.read_csv(
    'data/summarized_data/summarized_datacap.csv'
    , nrows=N_ROWS
)

def get_lgb_data(df_summarized):
    # These columns are created during summary, but not used for modeling
    drop_cols = [
        'product_name'
        ,'flight_id'
        ,'jack_seat_count'
        ,'total_usage_mb'
        ,'data_per_psn'
        ,'total_revenue_usd'
        ,'data_per_psn'
        ,'total_revenue_usd'
        ,'revenue_per_psn'
        ,'purchase_count'
        ,'takerate'
        ,'datacap'
        ,'timecap_min'
        ,'timecap'
        ,'price_per_min'
        ,'price_per_mb'
        ,'profit_per_psn'
    ]

    X = df_summarized.drop(columns=drop_cols)
    y = df_summarized['profit_per_psn']

    catcols = X.select_dtypes('object').columns
    X[catcols]  = X[catcols].astype('category')

    return {'data':X, 'target':y}

lgb_data = get_lgb_data(df_data)
print(lgb_data['data'].head())


#%%
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
#%%
X_train, X_test, y_train, y_test = train_test_split(lgb_data['data'], lgb_data['target'], test_size=0.2)

print('X train\n', X_train.dtypes)

print('\n\nX test\n', X_test.dtypes)


#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
lgb_reg = lgb.LGBMRegressor(
    random_state=SEED, 
    n_jobs=4,
    device=DEVICE,
    n_estimators=1000
    # max_bin=63
)

#This parameter defines the number of HP points to be tested
hyper_grid = {
    'num_leaves': np.arange(5, 300, 3),
    'min_child_samples': np.arange(100, 3000, 100),
    'min_child_weight': np.logspace(-5, 4, 10),
    'subsample': np.linspace(0.2, 1, 15),
    'colsample_bytree': np.linspace(0.45, 1, 15),
    'reg_lambda': np.logspace(-1, 3, 15)
}

# This passes additional args that are not in LGBMModel args
fit_params={
    'eval_set' : [(X_test,y_test)],
    'eval_names': ['valid'],
    'early_stopping_rounds':30,
    'categorical_feature': 'auto',
    'verbose':[1000],
    # 'device_type':['gpu']
}

rcv = RandomizedSearchCV(
    estimator=lgb_reg, 
    param_distributions=hyper_grid, 
    n_iter=N_HYPER,
    # n_jobs=1,
    cv=3,
    random_state=SEED,
    verbose=10
)

print(f'\n\n Fitting training data size: {X_train.shape}')

import time
start = time.time()
rcv = rcv.fit(X_train, y_train, **fit_params)
end = time.time()

print(f'Fitting took {round(end - start / 60)} minutes')


#%%
print(f'MAE train: {mean_absolute_error(y_train, rcv.predict(X_train))}')
print(f'RMSE train: {sqrt(mean_squared_error(y_train, rcv.predict(X_train)))}')
print(f'\nMAE validate: {mean_absolute_error(y_test, rcv.predict(X_test))}')
print(f'RMSE validate: {sqrt(mean_squared_error(y_test, rcv.predict(X_test)))} ')

print(f'Restults: {rcv.cv_results_}')
print(f'best parameters: {rcv.best_params_}')
print(f'best scores: {rcv.best_score_}')

from datetime import datetime
file = 'models/lgb_grid_random_' + datetime.now().strftime("%Y%m%d_%H_%M") + ".joblib"

joblib.dump(rcv,  file)

#%%
from sklearn.externals import joblib
rcv = joblib.load("models/lgb_grid_random_20190419_23_27.joblib")
scores = rcv.cv_results_['mean_test_score']
score_sorted = scores.argsort()
score_sorted

for i, ind in enumerate(score_sorted):
        print(i)
        print(rcv.cv_results_['params'][ind])

        if i > 5:
            break


#%%
