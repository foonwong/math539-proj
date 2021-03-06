import re
import pandas as pd
import json
import warnings 

def get_categorical_encodings(df, save_path=None):
    "Returns a dict of categorical encoding for future reference"

    catcols = df.select_dtypes('category').columns

    dict_cat = {}
    for col in catcols:
        encoding = {cat:lab for lab, cat in enumerate(df[col].cat.categories)}
        dict_cat[col] = encoding

    if save_path:
        json = json.dumps(dict_cat)
        f = open(save_path, 'w')
        f.write(json)
        f.close()

    return dict_cat

def preview(df, x=5):
    "Show first x and last x rows and dimensions of a dataframe"

    print(f'Size of dataframe: {df.shape}')

    out = pd.concat([df.head(x), df.tail(x)])

    return out


def distinct(df, cols, fillna=True):
    "similar to dplyr's distinct. Find distinct combination of a list of columns"

    out_df = df[cols].copy(deep=True)
    
    if fillna:
        for col in cols:
            if (out_df[col].dtype.name) == 'category':
                out_df.loc[:, col] = out_df[col].cat.add_categories([-999])

        out_df = out_df.fillna(-999)

    if sum(out_df.count() == out_df.shape[0]) < out_df.shape[0]:
        warnings.warn("Missing data is dropped")

    out_df = out_df.groupby(cols).size().reset_index().\
        drop(columns=0)

    return out_df


def move_target_to_end(df, target):
    "Move target column to the last column"
    colorder = [col for col in df.columns if col != target]
    colorder.append(target)

    return df[colorder]


def get_categorical_columns(df):
    cats = [col for col, typ in zip(df.columns, df.dtypes) 
        if re.search('category', typ.name)]

    return cats


def get_columns_by_type(df, types='.'):
    "Use regex pattern for multiple types"
    cats = [col for col, typ in zip(df.columns, df.dtypes) 
        if re.search(types, typ.name)]

    return cats


def missing_data_report(df):
    prop = 1 - round(df.count().sort_values() / df.shape[0], 4)
    prop = prop[prop > 0]

    print('\nMissing data report\n-------------------------------------')
    for i,v in zip(prop.index, prop):
        print(f'{i}:\t{v:.4%}')
    print(f'------------------------------\nDataframe dimensions: {df.shape}')

    return None


def is_subset(x, y):
    """Check if list x is subset of y 

    Keyword arguments:
    x, y: lists"""

    return set(x).issubset(set(y))
