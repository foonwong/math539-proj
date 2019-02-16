import pandas as pd

def data_reader(data, data_dict, nrows=None):
    ref = pd.read_csv(data_dict)
    type_dict = {k:v for k,v in zip(ref['col_name'], ref['pandas_type'])}

    df = pd.read_csv(data, dtype=type_dict, nrows=nrows)

    timecols = [x for x in df.columns if 'Time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    return df
