# def get_flight_revenue(df):
#     "Returns overall revenue per FlightID"
#     df_rev = df.groupby('flight_id')['price_usd'].sum().\
#         reset_index().rename(columns={'price_usd':'flight_revenue_usd'})

#     return df_rev

# def get_takerate_overall(df):
#     """Returns overall wifi takerate per FlightID"""
#     tr_df = df.\
#         groupby(['flight_id', 'total_passengers']).size().reset_index().rename(columns={0:'totalsessions'}).\
#         assign(flight_takerate = lambda x: x['total_sessions'] / x['total_passengers'])

#     return tr_df

# def get_flight_data_mb(df):
#     """Returns overall datausage per FlightID"""
#     df_du = df.groupby('flight_id')['total_usage_mb'].sum().\
#         reset_index().rename(columns={'total_usage_mb':'flight_data_mb'})

#     return df_du

# def get_flight_summary(df_detailed):
#     """Summarizing wifi dataframe for a flight level analysis"""
#     df = df_detailed.groupby('flight_id').size().reset_index().drop(columns=0)

#     try:
#         df_tr = get_takerate_overall(df_detailed)
#         df = pd.merge(df, df_tr, on=['flight_id', 'total_passengers'], how='left')
#     except:
#         pass

#     try:
#         df_du = get_flight_data_mb(df_detailed)
#         df = pd.merge(df, df_du, on=['flight_id'], how='left')
#     except:
#         pass

#     try:
#         df_rev = get_flight_revenue(df_detailed)
#         df = pd.merge(df, df_rev, on=['flight_id'], how='left')
#     except:
#         pass

#     return df