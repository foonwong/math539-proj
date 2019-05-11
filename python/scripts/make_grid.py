#%% [markdown]
### Making feature grid for pricing search 
# we need unique combos of categorical features + boolean
# other numerical features are aggregated to their median

import sys
sys.path.append('python')
import importlib
import re
import numpy as np
import pandas as pd
from wifipricing.modeling_prepartions import get_lgb_data


data_paths = {
    'datacap':'data/summarized_data/df_summarized_datacap_pickle.gzip',
    'timecap':'data/summarized_data/df_summarized_timecap_pickle.gzip',
    'fulldata':'data/summarized_data/df_summarized_all_pickle.gzip'
}


# for subset, path in data_paths.items():
#     df = pd.read_pickle(path)

#     X, y = get_lgb_data(df, subset)
#     X['y'] = y

#     outpath = re.sub("_pickle.*", ".feather", path)
#     X.to_feather(outpath)

# Somehow pandas run EXTREMELY slow for groupby.median()
# This can be accomplish in R in seconds


grid_paths = {
    'datacap':'data/summarized_data/df_summarized_datacap_featgrid.feather',
    'timecap':'data/summarized_data/df_summarized_timecap_featgrid.feather',
    'fulldata':'data/summarized_data/df_summarized_all_featgrid.feather'
}

for subset, path in grid_paths.items():
    df = pd.read_feather(path)
    print(df.dtypes)

pd.read_feather(grid_paths['datacap'])


