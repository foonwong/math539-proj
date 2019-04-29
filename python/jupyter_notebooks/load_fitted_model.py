#%% [markdown]
### A good way to check quantile model results

#%%
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt 

tm = '_23_46'

data = joblib.load(f'models/lgb_grid_data_20190428{tm}.joblib') 
models = {
    'lower': joblib.load(f'models/lgb_grid_quantile_lower_20190428{tm}.joblib')
    ,'median': joblib.load(f'models/lgb_grid_quantile_median_20190428{tm}.joblib')
    ,'upper': joblib.load(f'models/lgb_grid_quantile_upper_20190428{tm}.joblib')
}

#%%
for k, v in models.items():
    print('\n-----------------------------\n' + k + ':\n')
    print(f"MAE train: {mean_absolute_error(data['y_train'], v.predict(data['X_train']))}")
    print(f"RMSE train: {sqrt(mean_squared_error(data['y_train'], v.predict(data['X_train'])))}")
    print(f"\nMAE validate: {mean_absolute_error(data['y_test'], v.predict(data['X_test']))}")
    print(f"RMSE validate: {sqrt(mean_squared_error(data['y_test'], v.predict(data['X_test'])))} ")


#%%
import pandas as pd

#%%

df_data = pd.read_csv(
    'data/summarized_data/summarized_datacap.csv'
    , nrows=50
)

df_data2 = pd.read_csv(
    'data/summarized_data/summarized_timecap.csv'
    , nrows=50
)

df_data3 = pd.read_csv(
    'data/summarized_data/summarized_all.csv'
    , nrows=50
)


#%%
df_data.head()
#%%
df_data2.head()
#%%
df_data3.head()


#%%
