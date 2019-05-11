#%%
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open('models/predictions/datacap_predictions.pkl', 'rb') as pklFile:
    predictions = pickle.load(pklFile)

featgrid = predictions['feature_grid'].drop(columns=['dummy'])

for i, item in predictions.items():
    rec1  = item.sort_values(by=['y_median', 'ci_90'], ascending=[False,True])
    rec2  = item.sort_values(by=['y_lower', 'ci_90'], ascending=[False,True])
    rec3  = item.sort_values(by=['ci_90', 'y_median'], ascending=[True, False])

    print(f"\n\nRoute {i}: --------------------------")
    print(featgrid.iloc[[1]])
    print(f"For reference. The actual median for this feature is:", featgrid.iloc[[i]]['y'].values, '\n')
    print('\nSort by profit:')
    print(rec1.head())
    print('\nSort by lower bound:')
    print(rec2.head())
    print('\nSort by uncertainty:')
    print(rec3.head())

    if i > 3:
        break

#%%
sns.set()
sns.set_style("white")

def make_price_heatmap(pred_df, feat=None):
    print(feat)
    profit = pred_df.pivot("price_usd", "datacap_mb", "y_median")
    plt.figure(figsize=(10, 8))
    plt.title('Predicted profit per person')
    ax = sns.heatmap(profit)
    ax.invert_yaxis()

    return None

def make_ci_heatmap(pred_df, feat=None):
    print(feat)
    profit = pred_df.pivot("price_usd", "datacap_mb", "ci_90")
    plt.figure(figsize=(10, 8))
    plt.title('Prediction uncertainty')
    ax = sns.heatmap(profit, cmap="Blues")
    ax.invert_yaxis()

    return None

make_price_heatmap(predictions[1])
make_ci_heatmap(predictions[1])
featgrid.iloc[1]



#%%


#%%
