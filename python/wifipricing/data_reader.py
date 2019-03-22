import pandas as pd
import re

def data_reader(data, data_dict, nrows=None, usecols=None):
    """reading in wifi data as pandas.DataFrame"""

    ref = pd.read_csv(data_dict)
    type_dict = {k:v for k,v in zip(ref['col_name'], ref['pandas_dtype'])}

    df = pd.read_csv(data, dtype=type_dict, nrows=nrows, usecols=usecols)

    timecols = [x for x in df.columns if 'Time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    prod_nm = df['ProductName'].unique()
    datacap_dict = {x:get_datacap(x) for x in prod_nm}
    timecap_dict = {x:get_timecap(x) for x in prod_nm}

    df['DataCap_MB'] = df['ProductName'].map(lambda x: datacap_dict[x])
    df['TimeCap_min'] = df['ProductName'].map(lambda x: timecap_dict[x])

    return df


def get_datacap(x):
    """Extract data cap in MB from productname"""
    data = re.search('(?i)[\d]+(?= ?MB)', x)
    
    try:
        data =  float(data.group())
    except:
        data = None

    return data


def get_timecap(x):
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


def get_takerate(df):

    return None
