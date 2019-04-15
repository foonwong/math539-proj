#%%
sys.path.append('python')


#%%
import importlib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 15) 


#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader
from wifipricing.data_reader import get_flight_summary
from wifipricing.data_reader import get_product_summary
from wifipricing.data_reader import distinct
from wifipricing.data_reader import move_target_to_end


#%%
df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    # skiprows=np.random.choice(range(int(4e6)), int(3e6), replace=False), 
    usecols=[
        'flight_id', 
        'product_name', 
        'total_passengers', 
        'price_usd',
        'total_usage_mb', 
        'night_flight',
        # 'routes',
        'flight_duration_type',
        'orig_region',
        'dest_region',
        # 'orig_country',
        # 'dest_country',
        'economy_pct'
    ]
)

#%%
df = df.\
    assign(timecap = lambda x: x['timecap_min'] >= 0).\
    assign(datacap = lambda x: x['datacap_mb'] >= 0)


#%%
nrow = df.shape[0]
datacapped = df[df['datacap']].shape[0]
timecapped = df[df['timecap']].shape[0]
bothcapped = df[df['datacap'] & df['timecap']].shape[0]
nocapped = df[~(df['datacap'] | df['timecap'])].shape[0]

f'Data: {datacapped} '
f'time: {timecapped} '
f'both: {bothcapped} '
f'None: {nocapped} '


#%%
df_summarized_datacap = df[df['datacap']].pipe(get_product_summary)
df_summarized_timecap = df[df['timecap']].pipe(get_product_summary)

nocapcol = [x for x in df.columns if ('cap' not in x) and ('price_per' not in x)]
df_summarized_nocap = df[~(df['datacap'] | df['timecap'])][nocapcol].pipe(get_product_summary)



#%%
df_summarized_datacap.shape[0]
df_summarized_timecap.shape[0]
df_summarized_nocap.shape[0]

df_summarized_datacap.head(5)
df_summarized_timecap.head(5)
df_summarized_nocap.head(5)


#%%
p = sns.distplot(df_summarized_datacap['profit_per_psn'])

#%%
p = sns.distplot(df_summarized_timecap['profit_per_psn'])

#%%
p = sns.distplot(df_summarized_nocap['profit_per_psn'])


# #%%
# # Corr plot
# cols = ['price_per_mb',
#  'price_per_min',
#  'data_per_psn',
#  'revenue_per_psn',
#  'profit_per_psn',
#  'datacap_mb'] + list(['night_flight'])

# #%%
# g = sns.PairGrid(df_summarized_datacap[cols].sample(1000), hue='night_flight')
# g.map_lower(sns.kdeplot)
# g.map_upper(sns.scatterplot)
# g.map_diag(sns.kdeplot)


# #%%
# sns.distplot(prod_summary['profit_per_psn'])

# #%%
# prod_summary[['routes', 'profit_per_psn']].sort_values('profit_per_psn', ascending=False)

# #%%
# distinct(prod_summary, ['orig_country', 'orig_region'])

# #%%
# # prod_summary.groupby('routes').agg({'mean':'profit_per_psn'})
# prod_summary.groupby('orig_country')['profit_per_psn'].mean().reset_index().\
#     sort_values('profit_per_psn', ascending=False)
#     # query('orig_country.str.contains("Fin")')

# #%%
# prod_summary.groupby('dest_country')['profit_per_psn'].mean().reset_index().\
#     sort_values('profit_per_psn', ascending=False)
#     # query('orig_country.str.contains("Fin")')

# # prod_summary.groupby('dest_country')['profit_per_psn'].mean().reset_index().\
# #     sort_values('profit_per_psn', ascending=False)

# # sns.boxplot(x='routes', y='profit_per_psn', 
# #     data=prod_summary)


# #%%
# prod_summary.shape

# #%%
sns.scatterplot(
    x='price_usd', hue='datacap_mb',
    y='profit_per_psn',
    data=df_summarized_datacap.sample(5000)
)

#%%
mdl_cols = [
    'price_usd',
    'datacap_mb',
    # 'flight_duration_type',
    # 'orig_region',
    # 'dest_region',
    # 'night_flight',
    'economy_pct',
    # 'has_timecap',
    'profit_per_psn'
]

df_model = df_summarized_datacap.\
    assign(has_timecap = lambda x: x['price_per_min'].notnull()).\
    loc[:, mdl_cols].\
    pipe(move_target_to_end, target='profit_per_psn')

x = df_model.iloc[:, :-1]
y = df_model.iloc[:, -1]

#%%
x
y

#%%
# # This transforms everything!
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# x_transformed = ohe.fit_transform(x)


#%%
from sklearn.model_selection import KFold
kfold = KFold(n_splits =3)

for train_ind, test_ind in kfold.split(x):
    print("\nFOLD")
    print("train", train_ind)
    print("test", test_ind)

#%%
x.iloc[train_ind, :]
y[train_ind]

#%%
from lightgbm import LGBMRegressor
lgb_reg = LGBMRegressor(silent=False)

lgb_reg.fit(x.iloc[train_ind, :], y[train_ind])

prediction = lgb_reg.predict(x.iloc[test_ind, :])

p = sns.scatterplot(prediction, y[test_ind])
p.set(xlabel='prediction', ylabel='data', xlim=(-1,1), ylim=(-1,1))


#%%
from sklearn import linear_model
lasso = linear_model.Lasso()

lasso.fit(x.iloc[train_ind, :], y[train_ind])
prediction = lasso.predict(x.iloc[test_ind, :])
[(pred, y) for pred, y in zip(prediction, y[test_ind])]

p = sns.scatterplot(prediction, y[test_ind])
p.set(xlabel='prediction', ylabel='data', xlim=(-1,1), ylim=(-1,1))


#%%
