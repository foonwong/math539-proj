### RandomgridSearchCV to tune lgb for all three subset 
### 3 data subset x 3 quantiles = 9 total models fitted
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

N_HYPER = 200
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

data_paths = {
    'datacap':'data/summarized_data/df_summarized_datacap_pickle.gzip',
    'timecap':'data/summarized_data/df_summarized_timecap_pickle.gzip',
    'fulldata':'data/summarized_data/df_summarized_all_pickle.gzip'
}

# Outerloop: 3 data subset
for subset, path in data_paths.items():
    df = pd.read_pickle(path)

    X, y = get_lgb_data(df, subset)
    label_encoders = label_transform(X)

    # categorical_feature parameter for lgb
    # https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/sklearn.html
    cat_feat = list(label_encoders.keys())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('X train\n', X_train.dtypes)
    print('\n\nX test\n', X_test.dtypes)

    quantiles = {'lower': 0.05, 'median':0.5, 'upper':0.95}

    curtime = datetime.now().strftime("%Y%m%d_%H_%M")  
    start = time.time()
    # innerloop: 3 quantile regression alphas 
    for quantile, alp in quantiles.items():
        print(f'\n\n Fitting {quantile} quantile: {X_train.shape}\n')
        # Make sure that alpha is reset after each run
        seed=np.random.randint(0, 10000)
        lgb_reg = lgb.LGBMRegressor(random_state=seed, n_jobs=4,
                                    n_estimators=100, objective='quantile')

        # Validation
        rcv = lgb_random_search(
            X_train, X_test, y_train, y_test, lgb_reg, alp,
            HYPER_GRID, N_HYPER, seed, cat_feat, rmse, cv=5
        )

        y_validate = rcv.predict(X_test)

        # Refitting model with best param/full dataset 
        lgb_reg = lgb.LGBMRegressor(random_state=seed, n_jobs=4,
                                    n_estimators=1000, objective='quantile')
        lgb_reg.set_params(alpha = alp)
        lgb_reg.set_params(**rcv.best_params_)

        print(f'\n\n Refitting on full dataset with the following parameters:')
        print(lgb_reg.get_params())

        refit_params={
            'feature_name': list(X_train.columns),
            'categorical_feature': cat_feat
        }
        model = lgb_reg.fit(X, y, **refit_params)

        results = {'model': model,
                   'seed': seed,
                   'RMSE': rmse(y_test, y_validate)[1],
                   'y_test': y_test,
                   'y_predict': y_validate,
                   'X_sample': X_test.sample(n=1000),
                   'label_encoders': label_encoders,
                   'features': X_test.columns,
                   'training_data_size': X_train.shape,
                   'random_grid_cv': rcv}

        fnm = f'models/lgb_tuned_{subset}_{quantile}_quantile_{curtime}.joblib'
        joblib.dump(results, fnm)

    end = time.time()
    print(f'Fitting by random grid search for {subset} took {round((end - start) / 60)} minutes')
