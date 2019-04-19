#%%
N_HYPER = 10 
SEED = 1337
DEVICE = 'gpu'
N_JOBS = 4 
# N_ROWS= None
N_ROWS= 500000


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
import wifipricing.modeling_prepartions
importlib.reload(wifipricing.modeling_prepartions)
from wifipricing.modeling_prepartions import get_splitted_wifi_data 

#%%
df_data = pd.read_csv('data/summarized_data/summarized_datacap.csv')

#%%
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
        ,'profit_per_psn'
    ]

    X = df_summarized.drop(columns=drop_cols)
    y = df_summarized['profit_per_psn']

    # cats = X.select_dtypes(['category']).columns
    # nums = X.select_dtypes(['float64']).columns

    # cat_ref = dict()
    # print(f'\n* The {len(cats)} convert categorical to label encoding for lgb:') 
    # for c in cats:
    #     print(c)
    #     cat_ref[c] = X[c].cat.categories
    
    # X[cats] = X[cats].apply(lambda x: x.cat.codes)

    # print(f'\n* The {len(nums)} numerical columns will be centered and scaled:') 
    # for c in nums:
    #     print(c)

    return {'data':X, 'target':y}

lgb_data = get_lgb_data(df_data)

print(lgb_data)
# #%%
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from lightgbm import LGBMRegressor
# from scipy.stats import randint as sp_randint
# from scipy.stats import uniform as sp_uniform
# import lightgbm as lgb
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.externals import joblib
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from math import sqrt 

# X_train, X_test, y_train, y_test = train_test_split(lgb_data['data'], 
#                                                     lgb_data['target'],
#                                                     test_size=0.2)

# #This parameter defines the number of HP points to be tested
# hyper_grid ={'num_leaves': np.arange(5, 80, 10),
#              'min_child_samples': np.arange(300, 1500, 200),
#              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#              'subsample': np.linspace(0.2, 1, 15),
#              'colsample_bytree': np.linspace(0.45, 1, 15),
#              'reg_lambda': np.logspace(-1, 2, 10)}

# #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
# lgb_reg = lgb.LGBMRegressor(
#     random_state=SEED, 
#     silent=True,
#     n_jobs=N_JOBS, 
#     device=DEVICE, 
#     max_bin=63,
#     n_estimators=5000)

# lgb_randCV= RandomizedSearchCV(
#     estimator=lgb_reg, 
#     param_distributions=hyper_grid, 
#     n_iter=N_HYPER,
#     n_jobs=N_JOBS,
#     cv=3,
#     refit=True,
#     random_state=SEED,
#     verbose=100)

# fit_params={
#     "early_stopping_rounds":15, 
#     "eval_set" : [(X_test,y_test)],
#     'eval_names': ['valid'],
#     'categorical_feature': 'auto',
#     'verbose': -1 
# }

# print(f'\n\n Fitting training data size: {X_train.shape}')
# lgb_randCV.fit(X_train, y_train, **fit_params)

# #%%
# print(f'MAE train: {mean_absolute_error(y_train, lgb_randCV.predict(X_train))}')
# print(f'RMSE train: {sqrt(mean_squared_error(y_train, lgb_randCV.predict(X_train)))}')
# print(f'\nMAE validate: {mean_absolute_error(y_test, lgb_randCV.predict(X_test))}')
# print(f'RMSE validate: {sqrt(mean_squared_error(y_test, lgb_randCV.predict(X_test)))} ')

# print(f'Restults: {lgb_randCV.cv_results_}')
# print(f'best parameters: {lgb_randCV.best_params_}')
# print(f'best scores: {lgb_randCV.best_score_}')

# joblib.dump(lgb_randCV, 'models/lgb_grid_random.pkl')