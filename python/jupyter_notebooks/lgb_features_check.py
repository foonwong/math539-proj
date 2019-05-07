### Checking what features will be used for model tuning 
#%% [markdown]

## just a quick check to see what features are there and what will be used

#%%
import sys
sys.path.append('python')
import importlib
import re
import numpy as np
import pandas as pd
from wifipricing.modeling_prepartions import get_lgb_data
from wifipricing.modeling_prepartions import label_transform
from wifipricing.model_tuning import lgb_random_search
from wifipricing.model_tuning import rmse
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.externals import joblib
import time
from datetime import datetime

# N_HYPER = 200
# N_JOBS = 4
# N_ROWS= None
# HYPER_GRID = {
#     'num_leaves': np.arange(5, 300, 3),
#     'min_child_samples': np.arange(100, 3000, 100),
#     'min_child_weight': np.logspace(-5, 4, 10),
#     'subsample': np.linspace(0.2, 1, 15),
#     'colsample_bytree': np.linspace(0.45, 1, 15),
#     'reg_lambda': np.logspace(-1, 3, 15)
# }

data_paths = {
    'datacap':'data/summarized_data/df_summarized_datacap_pickle.gzip',
    'timecap':'data/summarized_data/df_summarized_timecap_pickle.gzip',
    'fulldata':'data/summarized_data/df_summarized_all_pickle.gzip'
}

for subset, path in data_paths.items():
    df = pd.read_pickle(path)

    print(f'\n-----------------------\nSubset {subset}:')
    print(f'* possible features:')
    print(df.columns)

    X, y = get_lgb_data(df, subset)

    print(f'\n* chosen features:')
    print(X.columns)




#%%
