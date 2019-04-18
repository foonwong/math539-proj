import pandas as pd
import re
from itertools import compress
from .data_cleaners import clean_wifi_data 
from .sugar import missing_data_report

def data_reader(data, data_dict, cleaning=True, nrows=None, skiprows=None, usecols=None):
    """reading in wifi data as pandas.DataFrame.

    Will create new columns for product data/time cap, pricing and profit
    if possible.

    Keyword arguments:
    data -- path to wifi csv
    data_dict -- path to reference csv with pandas colname and dtype info
    cleaning -- Clean and impute data. Reference data_cleaners
    nrows -- pd.read_csv nrows. Use None for all rows. (default None)
    skiprows -- pd.read_csv skiprows. 
    usecols -- pd.read_csv usecols. Use None for all cols. (default None)
    """

    ref = pd.read_csv(data_dict)

    pd_nm = ref['pandas_name']
    type_dict = {k:v for k,v in zip(pd_nm, ref['pandas_dtype'])}

    badcol = [x for x in usecols if x not in list(pd_nm)]

    if len(badcol) > 0: 
        raise Exception(f"{badcol} not a valid choice. Reference to refence's pandas_name")

    print(f'\nReading {data} with pd.read_csv()...')

    df = pd.read_csv(
        data,
        names = pd_nm, header=1,
        nrows=nrows, skiprows = skiprows, usecols=usecols,
        dtype = type_dict
    )

    timecols = [x for x in df.columns if 'time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    _flight_df_add_features(df)

    if cleaning:
        clean_wifi_data(df)

    missing_data_report(df)

    return df


def _flight_df_add_features(df):
    """Add extra features to wifi dataframe if possible
    (Modifies input dataframe)
    """

    print(f'\nAdding features\n------------------------------------')
    try:
        prod_nm = df['product_name'].unique()
    except:
        pass

    try:
        datacap_dict = {x:_get_datacap(x) for x in prod_nm}
        df['datacap_mb'] = df['product_name'].map(lambda x: datacap_dict[x])
        df['datacap'] = df['datacap_mb'].notnull()
        df['price_per_mb'] = df['price_usd'] / df['datacap_mb']
        print(f'Adding datacap_mb and price_usd')
    except:
        pass

    try:
        timecap_dict = {x:_get_timecap(x) for x in prod_nm}
        df['timecap_min'] = df['product_name'].map(lambda x: timecap_dict[x])
        df['timecap'] = df['timecap_min'].notnull()
        df['price_per_min'] =  df['price_usd'] / df['timecap_min']
        print(f'Adding timecap_min and price_min')
    except:
        pass

    try:
        df['profit'] = get_profit(df['price_usd'], df['total_usage_mb'])
        print(f'Adding profit (per session)')
    except:
        pass

    return None


def _get_datacap(x):
    """Extract data cap in MB from productname"""
    data = re.search('(?i)[\d]+(?= ?MB)', x)

    try:
        data =  float(data.group())
    except:
        data = None

    return data


def _get_timecap(x):
    """Extract time cap in min from productname"""
    data = re.search('(?i)([\d]+) ?(min|h)', x)

    try:
        if data.group(2).upper() == 'H':
            data =  60 * float(data.group(1))
        else:
            data = float(data.group(1))
    except:
        data = None

    return data