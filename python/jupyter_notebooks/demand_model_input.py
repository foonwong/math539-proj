#%%
import sys
import importlib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('python')
sns.set(color_codes=True)


#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 150) 


#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
import wifipricing.sugar 
importlib.reload(wifipricing.sugar)

from wifipricing.data_reader import data_reader
from wifipricing.data_reader import get_product_summary
from wifipricing.sugar import *


#%%
# Load data
usecol=[
    'product_name',
    'flight_id',
    'airline',
    'aircraft_type',
    'flight_duration_hrs',
    'total_usage_mb',
    'seat_count',
    'total_passengers',
    'night_flight',
    'flight_type',
    'category',
    'ac_frame',
    'routes',
    'price_usd',
    'ife',
    'e_xtv',
    'e_xphone',
    'one_media',
    'jack_seat_count',
    'economy_pct',
    'bus_pass_percent',
    'luxury',
    'origin_iata',
    'orig_country',
    'orig_region',
    # 'destination_iata',
    'dest_country',
    'dest_region'
    # 'airline_region'

    # 'antenna_type',
    # 'flight_duration_type'
    # 'session_start_time',
    # 'session_end_time',
    # 'session_duration_minutes',
    # 'session_volume_up_amount',
    # 'session_volume_down_amount',
    # 'departure_time_utc',
    # 'landing_time_utc',
    # 'departure_time_local',
    # 'landing_time_local'
]

df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    nrows=100000,
    # skiprows=np.random.randint(1, 4000000, 3500000),
    usecols=usecol
)

#%%
preview(df)
missing_data_report(df)


#%%


# #%%
# nrow = df.shape[0]
# datacapped = df[df['datacap']].shape[0]
# timecapped = df[df['timecap']].shape[0]
# bothcapped = df[df['datacap'] & df['timecap']].shape[0]
# nocapped = df[~(df['datacap'] | df['timecap'])].shape[0]

# f'Data: {datacapped} '
# f'time: {timecapped} '
# f'both: {bothcapped} '
# f'None: {nocapped} '


# #%%
# df['pass_seat_ratio'] = df['total_passengers'] / df['seat_count']
# df['pass_jack_ratio'] = df['total_passengers'] / df['jack_seat_count']

# df[df['pass_seat_ratio'] > 1][['aircraft_type', 'total_passengers', 'seat_count']].\
#     pipe(preview)



# #%%
# df_summarized_datacap = df[df['datacap']].pipe(get_product_summary)

# #%%
# df_summarized_timecap = df[df['timecap']].pipe(get_product_summary)

# nocapcol = [x for x in df.columns if ('cap' not in x) and ('price_per' not in x)]
# df_summarized_nocap = df[~(df['datacap'] | df['timecap'])][nocapcol].\
#     pipe(get_product_summary)




# #%%
# df_summarized_datacap.shape[0]
# df_summarized_timecap.shape[0]
# df_summarized_nocap.shape[0]

# #%%
# df_summarized_datacap.head(5)
# df_summarized_timecap.head(5)
# df_summarized_nocap.head(5)

# #%%
# df.groupby(['flight_id', 'product_name']).size().sort_values(ascending=False)

# #%%
# p = sns.distplot(df_summarized_datacap['profit_per_psn'])

# #%%
# p = sns.distplot(df_summarized_timecap['profit_per_psn'])

# #%%
# p = sns.distplot(df_summarized_nocap['profit_per_psn'])


# #%%
# sns.scatterplot(
#     x='price_usd', hue='datacap_mb',
#     y='profit_per_psn',
#     data=df_summarized_datacap.sample(5000)
# )

# #%%
# # Prepare for model: datacapped data
# mdlcols=[
#     # created columns
#     'datacap_mb'
#     , 'has_timecap'
#     , 'profit_per_psn'

#     # Original features
#     , 'airline'
#     , 'aircraft_type'
#     , 'flight_duration_hrs'
#     , 'seat_count'
#     , 'night_flight'
#     , 'flight_type'
#     , 'category'
#     , 'ac_frame'
#     # , 'routes'
#     , 'price_usd'
#     , 'ife'
#     , 'e_xtv'
#     , 'e_xphone'
#     , 'one_media'
#     , 'jack_seat_count'
#     , 'economy_pct'
#     , 'bus_pass_percent'
#     , 'luxury'

#     # , 'origin_iata'
#     , 'orig_country'
#     # , 'orig_region'
#     # , 'destination_iata'
#     , 'dest_country'
#     # , 'dest_region'
#     # , 'airline_region'

#     # , 'antenna_type'
#     # , 'flight_duration_type'
#     # , 'session_start_time'
#     # , 'session_end_time'
#     # , 'session_duration_minutes'
#     # , 'session_volume_up_amount'
#     # , 'session_volume_down_amount'
#     # , 'departure_time_utc'
#     # , 'landing_time_utc'
#     # , 'departure_time_local'
#     # , 'landing_time_local'
# ]

# f'Quick columns check: \
#     {[x for x in mdlcols if (x not in df_summarized_datacap.columns)]}\
#     is not in summary dataframe.'

# #%%
# df_model = df_summarized_datacap.\
#     assign(has_timecap = lambda x: x['price_per_min'].notnull()).\
#     loc[:, mdlcols].\
#     pipe(pd.get_dummies).\
#     pipe(move_target_to_end, target='profit_per_psn')

# df_model.head()
# f'model dataframe dimensions: {df_model.shape}'

# #%%
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# # col_encod = [c for c, t in zip(df_model.columns, df_model.dtypes) if t = 'category']
# # # col_scale = df_model.columns
# # col_encod

# df_model.columns
# df_model.dtypes


# #%%
# x = df_model.iloc[:, :-1]
# y = df_model.iloc[:, -1]

# x
# y



# #%%
# from sklearn.model_selection import KFold
# kfold = KFold(n_splits =3)

# for train_ind, test_ind in kfold.split(x):
#     print("\nFOLD")
#     print("train", train_ind)
#     print("test", test_ind)


# #%%
# x.iloc[train_ind, :]
# y[train_ind]


# #%%
# # LightGBM 
# from lightgbm import LGBMRegressor
# lgb_reg = LGBMRegressor(silent=False)

# lgb_reg.fit(x.iloc[train_ind, :], y[train_ind])

# prediction = lgb_reg.predict(x.iloc[test_ind, :])


# #%%
# feat_imp = lgb_reg.feature_importances_

# pd.DataFrame({'feat':x.columns, 'imp':feat_imp}).\
#     sort_values('imp', ascending=False).\
#     iloc[:20, :].\
#     plot(kind='bar', x='feat')


# #%%
# smp = np.random.randint(1, 90000, size = 10000)

# y_test = np.array(y[test_ind])

# sns.jointplot(prediction[smp], y_test[smp], kind='kde',
#     ylim = (0,1), xlim=(0,1))


# #%%
# # Good o random forest
# from sklearn.ensemble import RandomForestRegressor
# rf_reg = RandomForestRegressor()

# rf_reg.fit(x.iloc[train_ind, :], y[train_ind])

# prediction_rf = rf_reg.predict(x.iloc[test_ind, :])


# #%%
# feat_imp_rf = rf_reg.feature_importances_

# pd.DataFrame({'feat':x.columns, 'imp':feat_imp_rf}).\
#     sort_values('imp', ascending=False).\
#     iloc[:20, :].\
#     plot(kind='bar', x='feat')


# #%%
# smp = np.random.randint(1, 90000, size = 10000)

# y_test = np.array(y[test_ind])

# sns.jointplot(prediction_rf[smp], y_test[smp], kind='kde',
#     ylim = (0,1), xlim=(0,1))


# #%%
# #  Lasso
# from sklearn.linear_model import Lasso
# lasso_reg = Lasso(alpha=0.06)

# lasso_reg.fit(x.iloc[train_ind, :], y[train_ind])

# prediction_las = lasso_reg.predict(x.iloc[test_ind, :])


# #%%
# lasso_reg.coef_

# pd.DataFrame({'feat':x.columns, 'imp':lasso_reg.coef_}).\
#     sort_values('imp', ascending=False).\
#     iloc[:20, :].\
#     plot(kind='bar', x='feat')


# #%%
# smp = np.random.randint(1, 90000, size = 10000)

# y_test = np.array(y[test_ind])

# sns.jointplot(prediction_rf[smp], y_test[smp], kind='kde',
#     ylim = (0,1), xlim=(0,1))


# #%%
# # from sklearn import linear_model
# # from sklearn.model_selection import GridSearchCV

# # hyper_p = {'alpha': np.logspace(-6, 2, 10)}
# # print(hyper_p)

# # lasso = linear_model.Lasso(random_state=1337)
# # lassoCV = GridSearchCV(lasso, hyper_p)

# # lassoCV.fit(x, y)


# #%%
# # lassoCV.cv_results_

# # scores = lassoCV.cv_results_['mean_test_score']
# # scores_std = lassoCV.cv_results_['std_test_score']

# # plt.semilogx(hyper_p['alpha'], scores)

# # #%%

# # prediction = lasso.predict(x.iloc[test_ind, :])
# # [(pred, y) for pred, y in zip(prediction, y[test_ind])]

# # p = sns.scatterplot(prediction, y[test_ind])
# # p.set(xlabel='prediction', ylabel='data', xlim=(-1,1), ylim=(-1,1))


# # #%%
# # import re
# # # lassoCV.cv_results_[f'split{x}_test_score' for x in range(3alpha_)]
# # split_scores = [lassoCV.cv_results_[y] for y in lassoCV.cv_results_ if re.match('split.+test_score', y)]

# # plt.semilogx(hyper_p['alpha'], split_scores[1])

# # #%%
# # # Corr plot
# # cols = ['price_per_mb',
# #  'price_per_min',
# #  'data_per_psn',
# #  'revenue_per_psn',
# #  'profit_per_psn',
# #  'datacap_mb'] + list(['night_flight'])

# # #%%
# # g = sns.PairGrid(df_summarized_datacap[cols].sample(1000), hue='night_flight')
# # g.map_lower(sns.kdeplot)
# # g.map_upper(sns.scatterplot)
# # g.map_diag(sns.kdeplot)


# # #%%
# # sns.distplot(prod_summary['profit_per_psn'])

# # #%%
# # prod_summary[['routes', 'profit_per_psn']].sort_values('profit_per_psn', ascending=False)

# # #%%
# # distinct(prod_summary, ['orig_country', 'orig_region'])

# # #%%
# # # prod_summary.groupby('routes').agg({'mean':'profit_per_psn'})
# # prod_summary.groupby('orig_country')['profit_per_psn'].mean().reset_index().\
# #     sort_values('profit_per_psn', ascending=False)
# #     # query('orig_country.str.contains("Fin")')

# # #%%
# # prod_summary.groupby('dest_country')['profit_per_psn'].mean().reset_index().\
# #     sort_values('profit_per_psn', ascending=False)
# #     # query('orig_country.str.contains("Fin")')

# # # prod_summary.groupby('dest_country')['profit_per_psn'].mean().reset_index().\
# # #     sort_values('profit_per_psn', ascending=False)

# # # sns.boxplot(x='routes', y='profit_per_psn', 
# # #     data=prod_summary)


# # #%%
# # prod_summary.shape

# #%%
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# # col_encod = [c for c, t in zip(df_model.columns, df_model.dtypes) if t = 'category']
# # # col_scale = df_model.columns
# # col_encod

# # cats = get_categorical_columns(df_model) 
# #%%



# #%%
# df_model.one_media.dtype

# #%%
# testbool = data_reader('data/df_rfp_dataset_raw_20181218185047.csv',
#     data_dict='data/data_reference.csv',
#     nrows=100,
#     usecols=['ife', 'e_xtv', 'e_xphone', 'one_media'])

# preview(testbool)

# #%%
# testbool = pd.read_csv('data/df_rfp_dataset_raw_20181218185047.csv',
#     # data_dict='data/data_reference.csv',
#     nrows=50000, sep=',',
#     usecols=['IFE', 'eXTV', 'eXPhone', 'OneMedia'],
#     dtype={'IFE':'bool', 'e_xtv':'bool', 'e_xphone':'bool', 'one_media':'bool'})

# preview(testbool)

#%%

#%%