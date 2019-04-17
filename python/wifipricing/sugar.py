import re
import pandas as pd

def preview(df, x=5):
    "Show first x and last x rows and dimensions of a dataframe"

    print(f'Size of dataframe: {df.shape}')

    out = pd.concat([df.head(x), df.tail(x)])

    return out


def distinct(df, cols, fillna=True):
    "similar to dplyr's distinct. Find distinct combination of a list of columns"

    df = df[cols]

    if fillna:
        for col in cols:
            if (df[col].dtype.name) == 'category':
                df[col] = df[col].cat.add_categories([-999])

        df = df.fillna(-999)

    df = df.groupby(cols).size().reset_index().\
        drop(columns=0)

    return df


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