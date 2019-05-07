#%% [markdown]
## Example on how to use fitted objects
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt 
import os
import pandas as pd
import numpy as np

#%% [markdown]
### Loading fitted models

#%%
alljoblibs = os.listdir('models')
print(alljoblibs)

#%%
subset = {'datacap':{}, 'timecap':{},'fulldata':{}}

print('loading files double loop:') 
for sub in subset:
    joblibs = [file for file in alljoblibs if sub in file]

    for quantile in ['upper', 'median', 'lower']:
        path = [f'models/{x}' for x in joblibs if quantile in x][0]
        loaded = joblib.load(path)
        subset[sub].update({quantile: loaded})

        print(f'Loaded {sub}-{quantile} --- RMSE: {loaded["RMSE"]:.4f}')

print(subset.keys())
print(subset['datacap'].keys())



#%% [markdown]
### Let's say these are the rows that we want to apply model
dummy_data = subset['datacap']['median']['X_sample']
dummy_data = dummy_data.head(10)

dummy_data.index = np.arange(0, dummy_data.shape[0])  # resetting index from 0:nrows
dummy_data

#%%
predictions = {}
for quantile, fit_objs in subset['datacap'].items():
    model = fit_objs['model']

    predictions.update({f'y_predicted_{quantile}': model.predict(dummy_data)})

predictions = pd.DataFrame(predictions)
predictions

#%%
# Combining model predictions and data
predicted_df = pd.concat([predictions, dummy_data], axis=1)
predicted_df

#%% [markdown]
### Inverse label transformation
# The dummy data I used above has already been label transformed 
# because they are the actual traning data. 
# To return the actual categories, we need to use the label encoder saved
lab_encoder = subset['datacap']['median']['label_encoders']

for col, le in lab_encoder.items():
    inv_transformer = le.inverse_transform
    predicted_df[col] = predicted_df[col].transform(inv_transformer)

predicted_df


#%%
#%% [markdown]
### label transformation
# I'll need to write functions for this, but for now, the following should work

#%%
# new_df     # insert your own here
# new_df_encoded = new_df.copy()   # otherwise dataframes are modified by reference
# lab_encoder = subset['datacap']['median']['label_encoders']
# 
# for col, le in lab_encoder.items():
    # transformer = le.inverse_transform
    # new_df_encoded[col] = new_df_encoded[col].transform(transformer)