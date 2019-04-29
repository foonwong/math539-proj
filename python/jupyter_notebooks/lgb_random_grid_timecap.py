#%% [markdown]
### RandomgridSearchCV to tune lgb for timecap subset 

#%%
N_HYPER = 150
SEED = 1337
DEVICE = 'gpu'
N_JOBS = 4 
N_ROWS= None
# N_ROWS= 5000


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
    'data/summarized_data/summarized_timecap.csv'
    , nrows=N_ROWS
)

def get_lgb_data(df_summarized):
    # These columns are created during summary, but not used for modeling
    drop_cols = [
        'product_name'
        ,'flight_id'
        ,'airline'
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
        ,'datacap_mb'
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
    'verbose':[1000]
}

models = {'lower': 0.05, 'median':0.5, 'upper':0.95}

import time
start = time.time()

for k, alp in models.items():
    lgb_reg = lgb.LGBMRegressor(
        random_state=SEED, 
        n_jobs=4,
        device='gpu',
        objective='quantile',
        alpha = alp
    )

    rcv = RandomizedSearchCV(
        estimator=lgb_reg, 
        param_distributions=hyper_grid, 
        n_iter=N_HYPER,
        cv=5,
        random_state=SEED,
        verbose=15
    )

    print(f'\n\n Fitting {k} quantile: {X_train.shape}\n')

    models[k] = rcv.fit(X_train, y_train, **fit_params)

end = time.time()

print(f'Fitting took {round((end - start) / 60)} minutes')


from datetime import datetime
curtime = datetime.now().strftime("%Y%m%d_%H_%M")  
prefix = 'models/lgb_randgrid_timecap'

for k, mod in models.items():
    file = prefix + f'_quantile_{k}_' + curtime+ ".joblib"
    joblib.dump(mod,  file)

file = prefix + '_data_' + curtime+ ".joblib"
joblib.dump({'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}, file)

scores = rcv.cv_results_['mean_test_score']
score_sorted = scores.argsort()
score_sorted

for i, ind in enumerate(score_sorted):
        print(i)
        print(rcv.cv_results_['params'][ind])

        if i > 3:
            break