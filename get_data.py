from tqdm.auto import tqdm
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from src.dataset import Dataset
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel


from src.dataset import Dataset
from src.functions import remove_repeating
from src.functions import * 


time_deltas = [
    ("-1h", "-0h"),
    ("-2h", "-1h"),
    ("-8h", "-2h"),
    ("-24h", "-0h"),
    ("-24h", "24h"),
    ("0h", "1h"),
    ("1h", "2h"),
    ("2h", "8h"),
    ("0h", "24h"),
]

def compute_lag(df, deltas = time_deltas, index_inc = pd.Timedelta("15m"), relevant_index = lambda x: True):
    
    max_l_inc = int(max(abs(pd.Timedelta(l) / index_inc) for l, _ in deltas)) + 1
    max_u_inc = int(max(abs(pd.Timedelta(u) / index_inc) for _, u in deltas)) + 1
    
    columns = None
    new_data = []
    new_index = []
    for i, (index, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]):
        
        if not relevant_index is None and not relevant_index(index):
            continue
        
        local_data = []
        local_columns = []
        
        # Speed up by only selecting maximal relevant points
        local_range = df.iloc[max(i - max_l_inc, 0) : i + max_u_inc]
        local_range = local_range[local_range.index.map(lambda x: x.day == index.day)]
        
        # Add the new index
        new_index.append(index)
        
        for low, high in deltas:
            
            # Get data points
            start_date = index + pd.Timedelta(low)
            end_date = index + pd.Timedelta(high)
            
            # Compute stats
            temp =  local_range.loc[(local_range.index >= start_date) & (local_range.index <= end_date)]     
            
            stats = pd.concat([temp.mean().fillna(0), temp.std().fillna(0)])
            
            # Get names and build new df
            if columns is None:
                local_columns.extend([c + f"_[{low}, {high}]_mean" for c in temp.columns])
                local_columns.extend([c + f"_[{low}, {high}]_std"  for c in temp.columns])
            
            local_data.extend(stats.values.tolist())
    
        # Set up names
        if columns is None:
            columns = local_columns
    
        new_data.append(local_data)
    
    new_df = pd.DataFrame(new_data, columns=columns, index=new_index)

    return new_df

def impute_corrolation(*dfs, cols_to_impute):
    
    # Get full index
    idx = sum([c.index.tolist() for c in dfs], start=[])
    combined_index = pd.DatetimeIndex(set(idx))
    
    # Reindex to get all dates at all columns
    dfs_ = [c.reindex(combined_index, fill_value=np.nan) for c in dfs]
    
    for col in cols_to_impute:
    
        # Impute using mean - rows with all nans are not imputed on
        data = np.stack([getattr(df, col).values for df in dfs_], axis=1).transpose()
        mask = np.any(~np.isnan(data), axis=0)

        if np.any(mask):
            data[:, mask] = SimpleImputer().fit_transform(data[:, mask])
            data = data.transpose()

            # Update the output
            for i in range(len(dfs_)):      
                dfs_[i][col] = data[:, i]
                dfs[i][col] = dfs_[i][col].loc[dfs[i].index]
        
    return dfs

def load_basic(data):
    
    out = {}
    for key, values in data.items():
        
        # Get the local variables
        X_est, X_obs, X_test, y = (
            values[key] for key in [
                "X_train_estimated", "X_train_observed", 
                "X_test_estimated", "train_targets"
            ]
        )
        
        # Concat and align
        X_train = pd.concat([X_obs, X_est], axis = 0)
        
        # Set up indexing
        for X in [X_train, X_test]:
            X.set_index("date_forecast", inplace=True)
            X.drop(["date_calc"], inplace=True,axis=1)        
        y.set_index("time", inplace=True)

        out[key] = {
            "X_train": X_train,
            "y_train": y,
            "X_test": X_test
        }

    return out

def add_impute(data):

    keys = list(data.keys())

    # Impute using mean accors locations where possible
    for x_type in ["X_train", "X_test"]:
        impute_corrolation(
            *[data[key][x_type] for key in keys], 
            cols_to_impute=["ceiling_height_agl:m", "cloud_base_agl:m", "snow_density:kgm3"]
        )
    
    # Impute all missing using KNN
    for key in tqdm(keys):
        imputer = SimpleImputer()
        X_train = imputer.fit_transform(data[key]["X_train"])
        X_test = imputer.transform(data[key]["X_test"])

        data[key]["X_train"] = pd.DataFrame(X_train, columns=data[key]["X_train"].columns, index=data[key]["X_train"].index)
        data[key]["X_test"] = pd.DataFrame(X_test, columns=data[key]["X_test"].columns, index=data[key]["X_test"].index)

    return data


def add_lag(data):

    for key in tqdm(data.keys()):
          for x_type in ["X_train", "X_test"]:

                # Preprocess - only predic on full hours (min == 0)
                data[key][x_type] = compute_lag(
                    data[key][x_type], 
                    relevant_index = lambda idx: idx.minute == 0
                )

    return data
        

def add_remove(data):

    for key in tqdm(data.keys()):

        X_train = data[key]["X_train"]
        X_test = data[key]["X_test"]
        y = data[key]["y_train"]

        # Handle not aligned
        y_helper = y.copy()
        y_helper["time"] = y.index
        y_helper.reset_index(inplace=True, drop=True)

        y_helper = y_helper.dropna()
        y_helper = remove_repeating(y_helper)
        
        # Reset back
        y_helper.set_index("time", inplace=True)
        y = y_helper

        y = y[y.index.isin(X_train.index)]
        X_train = X_train[X_train.index.isin(y.index)]

        data[key] = {
            "X_train": X_train,
            "y_train": y,
            "X_test": X_test
        }
    
    return data

def add_selection(data):
    
    # Out data
    out = {}
    
    # Iterate over keys (locations)
    for key, _ in tqdm(data.items()):
        
        # Load data
        X_train = data[key]["X_train"]
        y = data[key]["y_train"]
        X_test = data[key]["X_test"]
        
        # Select relevant features
        estimator = GradientBoostingRegressor()
        selector = selector = SelectFromModel(estimator=estimator, max_features=40, threshold=None)
        X_train_ = selector.fit_transform(X_train, y)
        X_test_ = selector.transform(X_test)
        
        columns = [c for c, b in zip(X_train.columns, selector.get_support()) if b]

        print(f"Selected {len(columns)} features: ", columns, sep="\n")
        
        out[key] = {
            "X_train": pd.DataFrame(X_train_, columns=columns, index=X_train.index),
            "y_train": y,
            "X_test": pd.DataFrame(X_test_, columns=columns, index=X_test.index),
        }
    
    return out

if __name__ == "__main__":

    # First load
    # data = Dataset(splitter=None, preprocess=load_basic)
    # folder = Path("data_basic")
    # data.save(folder)
    
    # Impute
    # print("Begin impute")
    # data = Dataset(folder, splitter=None, preprocess=add_impute)
    # folder = Path("data_imputed")
    # data.save(folder)
    # print("Completed...")
    
    # # Lag
    # print("Begin lag")
    # data = Dataset(folder, splitter=None, preprocess=add_lag)
    # folder = Path("data_lag")
    # data.save(folder)
    # print("Completed...")
    
    # # Lag
    # print("Begin remove")
    # data = Dataset(folder, splitter=None, preprocess=add_remove)
    folder = Path("data_removed")
    # data.save(folder)
    # print("Completed...")
    
    # Lag
    print("Begin selection")
    data = Dataset(folder, splitter=None, preprocess=add_selection)
    folder = Path("data_selected")
    data.save(folder)
    print("Completed...")
