#%%
from itertools import product
import pandas as pd
import numpy as np
import sys
sys.path.append('python')
from wifipricing.modeling_prepartions import get_lgb_data
from sklearn.externals import joblib

#%% 
def get_price_grid(price_list, datacap_list):
    """returns dataframe price and datacap list combinations"""
    prod = product(price_list, datacap_list)
    p, d = zip(*prod)  #list of tuples to two tuples
    return pd.DataFrame({'price_usd':p, 'datacap_mb':d})


def label_encode_mapper(df, encoder_dict):
    '''Label transform a training dataframe with a dict of sklearn label encoder'''
    for col, encoder in encoder_dict.items():
        df[col] = encoder.transform(df[col])
    return None


def label_encode_invmapper(df, encoder_dict):
    '''Label inv_transform a training dataframe with a dict of sklearn label encoder'''
    for col, encoder in encoder_dict.items():
        df[col] = encoder.inverse_transform(df[col])
    return None


def newdata_transform(data, col_order, df_transformer, encoder_dict):
    newdata = data.copy()
    df_transformer(newdata, encoder_dict) 

    return newdata[col_order]


def prep_grids(fgrid, pgrid):
    fgrid.drop(columns=pgrid.columns, inplace=True)
    fgrid['dummy'] = 1
    pgrid['dummy'] = 1

    return None


def get_quantile_predictions(fit_dict, X_new):
    pred = {}
    for type, fit_obj in fit_dict.items():
        key = f'y_{type}'
        pred.update({key: fit_obj['model'].predict(X_new)})

    df_pred = pd.DataFrame(pred)
    df_pred['ci_90'] = df_pred['y_upper'] - df_pred['y_lower']


    return df_pred



#%% [markdown]
# Price grid settings
#%%
PRICES = list(range(51))
DATACAPS = list(np.arange(0, 160, 10)) 



#%% [markdown]
## Loading model fit objects + grids

#%%
alljoblibs = os.listdir('models')
subset = {
    'datacap':{}, 
    # 'timecap':{},
    # 'fulldata':{}
}

print('loading files double loop:') 
for sub in subset:
    joblibs = [file for file in alljoblibs if sub in file]

    for quantile in ['upper', 'median', 'lower']:
        path = [f'models/{x}' for x in joblibs if quantile in x][0]
        loaded = joblib.load(path)
        subset[sub].update({quantile: loaded})

        print(f'Loaded {sub}-{quantile} --- RMSE: {loaded["RMSE"]:.4f}')

datacap_fits = subset['datacap']
encoder_dict = datacap_fits['median']['label_encoders']

grid_paths = {
    'datacap':'data/summarized_data/df_summarized_datacap_featgrid.feather',
    # 'timecap':'data/summarized_data/df_summarized_timecap_featgrid.feather',
    # 'fulldata':'data/summarized_data/df_summarized_all_featgrid.feather'
}

# for subset, path in grid_paths.items():
#     df = pd.read_feather(path)

featgrid = pd.read_feather(grid_paths['datacap'])
price_grid = get_price_grid(PRICES, DATACAPS)

prep_grids(featgrid, price_grid)

#%%
featgrid.head()
#%%
price_grid.head()

#%% [markdown]
### Original training data shape reference:
# We need to make sure that our generated data has the same dtype and order

#%%
datacap_fits['median']['X_sample'].head()


#%% [markdown]
## Iterate over grids
#%%
import time
from tqdm import tqdm

model_col_order = datacap_fits['median']['X_sample'].columns

predictions = {}
predictions.update({'feature_grid': featgrid})

for i, df_row in tqdm(featgrid.groupby(level=0)):
    row_pricegrid = pd.merge(df_row, price_grid, on='dummy').drop(columns=['dummy'])
    X_new = newdata_transform(row_pricegrid, model_col_order, label_encode_mapper, encoder_dict)

    pred = get_quantile_predictions(datacap_fits, X_new)
    predictions[i] = pd.concat([price_grid.drop(columns='dummy'), pred], axis = 1)

import pickle
with open('models/predictions/datacap_predictions.pkl', 'wb') as pklFile:
    pickle.dump(predictions, pklFile)

# for i, item in predictions.items():
#     rec1  = item.sort_values(by=['y_median', 'ci_90'], ascending=[False,True])
#     rec2  = item.sort_values(by=['y_lower', 'ci_90'], ascending=[False,True])
#     rec3  = item.sort_values(by=['ci_90', 'y_median'], ascending=[True, False])

#     print(f"\n\nRoute {i}: --------------------------")
#     print(df_row)
#     print(f"For reference. The actual median for this feature is:", df_row['y'].values, '\n')
#     print('\nSort by profit:')
#     print(rec1.head())
#     print('\nSort by lower bound:')
#     print(rec2.head())
#     print('\nSort by uncertainty:')
#     print(rec3.head())

#     if i > 30:
#         break


