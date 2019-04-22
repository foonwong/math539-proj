#%%
# N_HYPER = 30 
SEED = 1337
DEVICE = 'gpu'
N_JOBS = 4 
N_ROWS= None
# N_ROWS= 1000


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
    n_jobs=N_JOBS, 
    device=DEVICE, 
    n_estimators=1000
)

def LGB_RandCV(params, n_hyper):
    rcv = RandomizedSearchCV(
        estimator=lgb_reg, 
        param_distributions=params, 
        n_iter=n_hyper,
        n_jobs=1,
        cv=3,
        random_state=SEED,
        verbose=20
    )

    return rcv

def LGB_GridCV(params):
    gcv = GridSearchCV(
        estimator=lgb_reg, 
        param_grid=params, 
        n_jobs=1,
        cv=3,
        verbose=20
    )

    return gcv

#This parameter defines the number of HP points to be tested
hyper_grid_1 = {
    'num_leaves': np.arange(3, 20, 1),
    'min_child_samples': np.arange(300, 3000, 20),
}

hyper_grid_2 = {
    'min_child_weight': np.logspace(-5, 4, 50)
}

hyper_grid_3 = {
    'subsample': np.linspace(0.2, 1, 15),
    'colsample_bytree': np.linspace(0.40, 1, 15),
    'reg_lambda': np.logspace(-2, 3, 40)
}


# This passes additional args that are not in LGBMModel args
fit_params={
    'eval_set' : [(X_test,y_test)],
    'eval_names': ['valid'],
    'early_stopping_rounds':20,
    'categorical_feature': 'auto',
    'verbose':[1000]
}

print(f'\n\n Fitting training data size: {X_train.shape}')

import time
start = time.time()
print(f'\n\n Round 1: {hyper_grid_1}')
lgb_tune_1 = LGB_RandCV(hyper_grid_1,  100).fit(X_train, y_train, **fit_params)

for k, y in lgb_tune_1.best_params_.items():
    hyper_grid_2.update({k: [y]})

print(f'\n\n Round 2: {hyper_grid_2}')
lgb_tune_2 = LGB_GridCV(hyper_grid_2).fit(X_train, y_train, **fit_params)

for k, y in lgb_tune_2.best_params_.items():
    hyper_grid_3.update({k: [y]})

print(f'\n\n Round 3: {hyper_grid_3}')
lgb_tune_3 = LGB_RandCV(hyper_grid_3,  100).fit(X_train, y_train, **fit_params)
end = time.time()
print(f'Fitting took {round(end - start / 60)} minutes')

#%%
print(f'MAE train: {mean_absolute_error(y_train, lgb_tune_3.predict(X_train))}')
print(f'RMSE train: {sqrt(mean_squared_error(y_train, lgb_tune_3.predict(X_train)))}')
print(f'\nMAE validate: {mean_absolute_error(y_test, lgb_tune_3.predict(X_test))}')
print(f'RMSE validate: {sqrt(mean_squared_error(y_test, lgb_tune_3.predict(X_test)))} ')


from datetime import datetime
from sklearn.externals import joblib
file = 'models/lgb_grid_iterative' + datetime.now().strftime("%Y%m%d_%H_%M") + ".joblib"
joblib.dump(lgb_tune_3,  file)

scores = lgb_tune_3.cv_results_['mean_test_score']
score_sorted = scores.argsort()
score_sorted

for i, ind in enumerate(score_sorted):
        print(f"\nParams {i}\n", lgb_tune_3.cv_results_['params'][ind])
        print('Mean test score: ' ,lgb_tune_3.cv_results_['mean_test_score'][ind])

        if i > 5:
            break
