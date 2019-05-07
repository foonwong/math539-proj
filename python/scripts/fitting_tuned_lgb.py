#%%
N_HYPER = 10 
SEED = 1337
DEVICE = 'gpu'
N_JOBS = 4 
N_ROWS= None
# N_ROWS= 50000


#%%
import sys
import importlib
import re
import numpy as np
import pandas as pd

#%%
# Notebook options
sys.path.append('python')
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%%
lgb_data = pd.read_csv('data/summarized_data/summarized_all.csv', nrows=N_ROWS)
# lgb_data = pd.read_csv('data/summarized_data/summarized_datacap.csv', nrows=N_ROWS)

def get_lgb_data(df_summarized):
    # These columns are created during summary, but not used for modeling
    drop_cols = [
        'product_name'
        ,'total_passengers'
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
        # ,'datacap'
        ,'datacap_mb'
        ,'timecap_min'
        # ,'timecap'
        ,'price_per_min'
        ,'price_per_mb'
        ,'price_usd'
        ,'profit_per_psn'
    ]

    cats = ['airline', 'orig_country', 'flight_type', 'dest_country']
    df_summarized[cats] = df_summarized[cats].astype('category')

    X = df_summarized.drop(columns=drop_cols)
    y = df_summarized['profit_per_psn']

    return {'data':X, 'target':y}

lgb_data = get_lgb_data(lgb_data)

lgb_data['data']


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

X_train, X_test, y_train, y_test = train_test_split(lgb_data['data'], 
                                                    lgb_data['target'],
                                                    test_size=0.2)

print(X_train.columns)

opt_param = {
    'subsample': 0.3142857142857143, 
    'reg_lambda': 0.21544346900318834, 
    'num_leaves': 15, 
    'min_child_weight': 10.0, 
    'min_child_samples': 300, 
    'colsample_bytree': 0.6071428571428572
} 

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
lgb_reg = lgb.LGBMRegressor(
    random_state=SEED, 
    silent=True,
    n_jobs=N_JOBS, 
    device=DEVICE, 
    max_bin=255,
    n_estimators=5000,
    **opt_param)

fit_params={
    "early_stopping_rounds":50, 
    "eval_set" : [(X_test,y_test)],
    'eval_names': ['valid'],
    'verbose': 500,
    'categorical_feature': 'auto'
}

lgb_reg.fit(X_train, y_train, **fit_params)

#%%
print(f'MAE train: {mean_absolute_error(y_train, lgb_reg.predict(X_train))}')
print(f'RMSE train: {sqrt(mean_squared_error(y_train, lgb_reg.predict(X_train)))}')
print(f'\nMAE validate: {mean_absolute_error(y_test, lgb_reg.predict(X_test))}')
print(f'RMSE validate: {sqrt(mean_squared_error(y_test, lgb_reg.predict(X_test)))} ')

joblib.dump(lgb_reg, 'models/lgb_optimized.pkl')

#%%
lgb_reg.n_features_
lgb_reg.objective_
lgb_reg.get_params
lgb_reg.feature_importances_

#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('pastel')
lgb.plot_importance(lgb_reg, figsize=(6, 8))
lgb.plot_metric(lgb_reg, figsize=(6, 8))

#%%



import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%%
