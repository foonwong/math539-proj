#%%
import sys
import os
import importlib
import re
import numpy as np
import pandas as pd

sys.path.append('python')


#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 


#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
import wifipricing.modeling_prepartions
importlib.reload(wifipricing.modeling_prepartions)
import wifipricing.sugar 
importlib.reload(wifipricing.sugar)

from wifipricing.data_reader import data_reader
from wifipricing.modeling_prepartions import get_splitted_wifi_data 
from wifipricing.sugar import *

#%%
df_data, df_time, df_none, df_all = get_splitted_wifi_data(
    'data/df_rfp_dataset_raw_20181218185047.csv', 
    'data/data_reference.csv' 
)

df_data.shape
df_time.shape
df_none.shape
df_all.shape



#%%
from sklearn.preprocessing import StandardScaler

def get_lgb_data(df_summarized):
    # These columns are created during summary, but not used for modeling
    drop_cols = ['product_name', 'flight_id', 'data_per_psn', 'revenue_per_psn', 'total_passengers', 'profit_per_psn']

    X = df_summarized.drop(columns=drop_cols)
    y = df_summarized['profit_per_psn']

    cats = X.select_dtypes(['category']).columns
    nums = X.select_dtypes(['float64']).columns

    cat_ref = dict()
    print(f'\n* The {len(cats)} convert categorical to label encoding for lgb:') 
    for c in cats:
        print(c)
        cat_ref[c] = X[c].cat.categories
    
    X[cats] = X[cats].apply(lambda x: x.cat.codes)

    print(f'\n* The {len(nums)} numerical columns will be centered and scaled:') 
    for c in nums:
        print(c)

    scaler = StandardScaler()
    X[nums] = scaler.fit_transform(X[nums])

    return {'data':X, 'target':y, 'labels':cat_ref, 'categorical':cats}

lgb_data = get_lgb_data(df_data)

lgb_data['data']
lgb_data['target']
lgb_data['categorical']


#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

X_train, X_test, y_train, y_test = train_test_split(lgb_data['data'], 
                                                    lgb_data['target'],
                                                    test_size=0.2)

# Tuning lgb:
seed = 1337

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'nthread': 3, # Updated from nthread
    'learning_rate': 0.05,
    'subsample_for_bin': 200,
    'subsample': 1,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 5,
    'reg_lambda': 10,
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 5,
    'scale_pos_weight': 1,
    'num_class' : 1,
    'metric' : 'binary_error'
}

gridParams = {
    'learning_rate': [0.005],
    'n_estimators': [40],
    'max_depth': [3, 5, 7],
    'num_leaves': [60, 70, 80],
    'min_data_in_leaf': [500, 1000, 1500],
    'random_state' : [seed], 
    'bagging_fraction' : np.logspace(-1, 0, 3),
    'feature_fraction' : np.logspace(-1, 0, 3), 
    'lambda_l2' : np.logspace(-3, 2, 10) 
}

lgb_reg = LGBMRegressor(task='train',
    boosting_type= 'gbdt',
    n_jobs = 4, 
    subsample_for_bin = params['subsample_for_bin'],
    subsample = params['subsample'],
    subsample_freq = params['subsample_freq'],
    min_split_gain = params['min_split_gain'],
    min_child_weight = params['min_child_weight'],
    min_child_samples = params['min_child_samples'],
    scale_pos_weight = params['scale_pos_weight'],
    device='gpu'
)

lgb_grid = GridSearchCV(lgb_reg, gridParams,
    verbose=15,
    cv=4, n_jobs=3
)

lgb_grid

#%%
lgb_grid.fit(X_train, y_train)


#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f'MAE train: {mean_absolute_error(y_train, lgb_grid.predict(X_train))}')
print(f'MSE train: {mean_squared_error(y_train, lgb_grid.predict(X_train))}')

print(f'\nMAE validate: {mean_absolute_error(y_test, lgb_grid.predict(X_test))}')
print(f'MSE validate: {mean_squared_error(y_test, lgb_grid.predict(X_test))} ')

#%%
from sklearn.externals import joblib
joblib.dump(lgb_grid, 'lgb_grid_dummy.pkl')