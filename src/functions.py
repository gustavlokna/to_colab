import numpy as np
import pandas as pd
import operator 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from src.dataset import *

        
def remove_repeating(df):
    # Assuming df is your DataFrame and y_c is defined elsewhere

    df["pv_measurement"] = df["pv_measurement"].abs()

    # Convert 'time' column to datetime if it's not already
    # Add a helper column to identify shifts in 'pv_measurement' value and same day check
    df['shifted_pv'] = df['pv_measurement'].shift().bfill()
    df['same_day'] = df['time'].dt.date == df['time'].shift().dt.date

    # Identify rows where the value is the same as the previous one and is on the same day
    df['same_as_prev'] = (df['pv_measurement'] == df['shifted_pv']) & df['same_day']

    # Create a group identifier for consecutive identical measurements within the same day
    df['group'] = (~df['same_as_prev']).cumsum()

    # Prepare a DataFrame to collect periods to remove
    indices_to_remove = []

    # Process each group and determine if it should be removed
    for name, group in df.groupby(['group', df['time'].dt.date]):
        duration_hours = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds() / 3600
        duration_count = len(group)
        value = group['pv_measurement'].iloc[0]
        if value != 0 and duration_hours > 3:
            # For non-zero values repeated for more than 3 hours
            indices_to_remove.extend(group.index)
        elif value == 0 and duration_count == 24:
            # For zero values repeated for all 24 hours of the same day
            indices_to_remove.extend(group.index)

    # Create a new DataFrame without the repeated periods
    df_cleaned = df.drop(indices_to_remove)

    # Drop the helper columns from the new DataFrame
    df_cleaned.drop(['shifted_pv', 'same_day', 'same_as_prev', 'group'], axis=1, inplace=True)
    return df_cleaned

def drop_repeating_sequences_and_return_y_with_droped_indixes(df):
    indexes_to_drop = set()  # Change this to a set to avoid duplicates
    prev_val = None
    consecutive_count = 0
    y_with_indexes_to_drop = df.copy()

    for i, val in enumerate(df["pv_measurement"]):
        if val != 0:
            if val == prev_val:
                consecutive_count += 1
            else:
                prev_val = val
                consecutive_count = 0

            if consecutive_count >= 1:
                indexes_to_drop.add(i - consecutive_count)  # Add to set to ensure uniqueness
                indexes_to_drop.add(i)  # Add to set to ensure uniqueness
    
    # Convert the set to a sorted list to use for indexing
    indexes_to_drop = sorted(indexes_to_drop)
    
    # Create the DataFrame without the dropped indices
    df_without_dropped = df.drop(indexes_to_drop)
    
    # Create the DataFrame with only the dropped indices
    df_with_only_dropped = df.loc[indexes_to_drop]

    return df_without_dropped, df_with_only_dropped


def delete_ranges_of_zeros_and_interrupting_values_and_return_y_with_dropped_indices(df, number_of_recurring_zeros, interrupting_values=[]):
    count = 0
    drop_indices = []

    # Store the original DataFrame to reinsert NaN values later
    original_df = df.copy()

    # Get the indices of the NaN values
    nan_indices = df[df['pv_measurement'].isna()].index.tolist()
    # Drop the NaN values for processing zeros and interrupting values
    df = df.dropna()

    for index, row in df.iterrows():
        if row["pv_measurement"] == 0 or row["pv_measurement"] in interrupting_values:
            count += 1
        else:
            if count > number_of_recurring_zeros:
                drop_indices.extend(df.index[index - count:index])
            count = 0

    if count > number_of_recurring_zeros:
        drop_indices.extend(df.index[index - count + 1:index + 1])
    
      # Convert the set to a sorted list to use for indexing
    indexes_to_drop = sorted(drop_indices)
    
    # Create the DataFrame without the dropped indices
    df_without_dropped = df.drop(indexes_to_drop)
    
    # Combine drop_indices with nan_indices to get all indices to be dropped
    all_drop_indices = sorted(set(drop_indices + nan_indices))

    # Create the DataFrame with only the dropped indices
    df_with_only_dropped = original_df.loc[all_drop_indices]
    
    return df_without_dropped, df_with_only_dropped  #, df_with_only_dropped (uncomment if needed)

def drop_long_sequences_and_return_y_with_dropped_indices(df, x):
    indexes_to_drop = []
    zero_count = 0

    for i, val in enumerate(df['pv_measurement']):
        if val == 0:
            zero_count += 1
        else:
            if zero_count >= x:
                start_index = i - zero_count
                end_index = i - 1  # inclusive
                if start_index >= 0 and end_index < len(df):
                    indexes_to_drop.extend(list(range(start_index, end_index + 1)))
            zero_count = 0

    # In case the sequence ends with zeros, this will handle it
    if zero_count >= x:
        start_index = len(df) - zero_count
        end_index = len(df) - 1
        if start_index >= 0 and end_index < len(df):
            indexes_to_drop.extend(list(range(start_index, end_index + 1)))

    # Create a dataframe with only the dropped rows
    df_with_only_dropped = df.loc[df.index[indexes_to_drop]].copy()

    # Drop the rows from the original dataframe
    df_dropped = df.drop(df.index[indexes_to_drop])
    
    return df_dropped, df_with_only_dropped



def mean_df(df):
    # Assuming df is your DataFrame and 'date_forecast' is your date column
    # Making a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Step 1: Keeping every 4th row in the date column
    date_column = df_copy['date_forecast'].iloc[::4]
    
    #Made it such that all the intresting columns having data for 1 hour average back in time is saved such for the last hour. Ex diffuse_rad_1h:j cl. 23:00 is used for the weather prediction 22:00
    selected_col = ['diffuse_rad_1h:J', 'direct_rad_1h:J', 'clear_sky_energy_1h:J']
    selected_values = df_copy[selected_col].iloc[4::4].reset_index(drop=True)
    last_row = pd.DataFrame(df_copy[selected_col].iloc[-1]).T.reset_index(drop=True)
    selected_values = pd.concat([selected_values, last_row], ignore_index=True)
    
    # Step 2: Creating a grouping key
    grouping_key = np.floor(np.arange(len(df_copy)) / 4)
    
    # Step 3: Group by the key and calculate the mean, excluding the date column
    averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()
    # Step 4: Reset index and merge the date column
    averaged_data.reset_index(drop=True, inplace=True)
    averaged_data['date_forecast'] = date_column.values
    averaged_data[selected_col] = selected_values.values
    return averaged_data



def agumenting_time(df):
    df["new_time"] = pd.to_datetime(df['date_forecast'])
    df['hour'] = df['new_time'].dt.hour
    df['minute'] = df['new_time'].dt.minute
    #df["day"]  = df['new_time'].dt.day
    df["month"]  = df['new_time'].dt.month
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    phase_adjustment = (np.pi/2) - 11 * (2 * np.pi / 24)
    df['hour_sin'] = np.sin(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    df['hour_cos'] = np.cos(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    df = df.drop(columns = ["new_time"])
    return df

def direct_rad_div_diffuse_rad(df):
    df['dif_dat_rad'] = 0.0
    condition = df['diffuse_rad:W'] != 0
    df.loc[condition, 'dif_dat_rad'] = df.loc[condition, 'direct_rad:W'] / df.loc[condition, 'diffuse_rad:W']
    return df


def add_all_features(df):
    df = direct_rad_div_diffuse_rad(df)
    df = agumenting_time(df)
    return df

def df_feature_selection(df, selected_features):
    return df[selected_features]

def resize_training_data(X_train, y_train):
    y_features = y_train.columns.tolist()
    X_date_feature = "date_forecast"
    
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized


def subset_months(df, wanted_months):
    df["month"]  = df['date_forecast'].dt.month
    df_subset = df[df["month"].isin(wanted_months)]
    return df_subset

def hyperparameter_tuning_old(param_grid, X_train, y_train):
    # Create a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5)

    # Fit the grid search
    grid_search.fit(X_train, y_train["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params}")
    return best_params

"""

def preprocess(data):
    
    times_y = set(data["train_targets"].time)
    
    for key in ["X_train_estimated", "X_train_observed"]:
            idx_X = data[key].date_forecast.isin(times_y).values
            data[key] = data[key][idx_X]
    return data

def hyperparameter_tuning(est, param_grid, data):

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    # Create the grid search
    grid_search = GridSearchCV(
        estimator=est, param_grid=param_grid, scoring=scorer,
        #cv = data.split_index() 
        cv=5
    )
    
    # Should be done in pipeline
    X = data.X.dropna(axis=1)
    X = X[[c for c in X.columns if not c in ['date_calc', 'date_forecast']]]
    #print(type(X))
    #print(type(data.y["pv_measurement"]))
    # Fir the grid search
    grid_search.fit(X, data.y["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    return best_params
    
    
"""

def clean_mean_combine(X_observed, X_estimated, selected_features):
    X_observed_clean = clean_df(X_observed, selected_features)
    X_estimated_clean = clean_df(X_estimated, selected_features)
    X_estimated_clean_mean = mean_df(X_estimated_clean)
    X_observed_clean_mean = mean_df(X_observed_clean)
    X_train = pd.concat([X_observed_clean_mean, X_estimated_clean_mean])
    X_train = add_lag_and_lead_features(X_train, 1, ["direct_plus_diffuse","direct_plus_diffuse_1h"])
    X_train = rolling_aggregates(X_train.copy(), rolling_aggregates_list, 24)
    return X_train



def rolling_aggregates(df, cols, window):
    for col in cols:
        df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
        df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
        df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
        df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
    return df

def resample_to_quarterly(df, interpolation_method, poly_order = 0):
    # Set the time column as the index
    df.set_index('time', inplace=True)
    
    if interpolation_method == "polynomial":
        # Resample to 15-minute intervals and interpolate
        df_quarterly = df.resample('15T').interpolate(method=interpolation_method, order = poly_order)
    else:
        df_quarterly = df.resample('15T').interpolate(method=interpolation_method)
    
    # Reset the index
    df_quarterly.reset_index(inplace=True)
    return df_quarterly

def create_new_feature(df, column1, column2, operation):
    # Define a dictionary mapping strings to actual Python operator functions
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }
    
    # Check if the operation is valid
    if operation not in ops:
        raise ValueError("Operation must be one of '+', '-', '*', '/'")
    
    # Perform the operation and create a new column
    new_column_name = f"{column1}{operation}{column2}"
    df[new_column_name] = ops[operation](df[column1], df[column2])
    df.replace([np.inf, -np.inf], np.nan, inplace=True).dropna(-1)
    
    return df
    
    
    
    
    
def add_all_features(df):
    df = direct_rad_div_diffuse_rad(df)
    df = agumenting_time(df)
    df["direct_plus_diffuse"] = df["direct_rad:W"] + df["diffuse_rad:W"]
    df["direct_plus_diffuse_1h"] = df["direct_rad_1h:J"] + df["diffuse_rad_1h:J"]
    return df

def rolling_aggregates(df, cols, window):
    for col in cols:
        df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
        df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
        df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
        df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
    return df


def add_lag_and_lead_features(df, lag_steps=1, columns = []):
    # Create a new DataFrame to hold the lagged and lead features
    lagged_df = pd.DataFrame()

    # Make sure the 'date' column is a datetime type
    df['date_forecast'] = pd.to_datetime(df['date_forecast'])

    # Group by date to ensure continuity within each day
    grouped = df.groupby(df['date_forecast'].dt.date)

    for _, group in grouped:
        # Reset index to allow proper shifting within group
        group = group.reset_index(drop=True)

        # Copy the current group to avoid modifying the original data
        temp_group = group.copy()

        # Iterate over all columns to create lagged and lead versions
        for column in columns:
            # Skip the date column if it exists
            if column == 'date' or column == 'date_forecast':
                continue

            # Create lagged feature for previous values
            lagged_column_name = f"{column}_lag{lag_steps}"
            temp_group[lagged_column_name] = group[column].shift(lag_steps).fillna(group[column])

            # Create lead feature for future values
            lead_column_name = f"{column}_lead{lag_steps}"
            temp_group[lead_column_name] = group[column].shift(-lag_steps).fillna(group[column])

            # Create a column for the difference between the lagged value and the present value (lag -1 - present)
            diff_lag_column_name = f"{column}_diff_lag{lag_steps}"
            temp_group[diff_lag_column_name] = temp_group[lagged_column_name] - group[column]

            # Create a column for the difference between the lead value and the present value (lag +1 - present)
            #diff_lead_column_name = f"{column}_diff_lead{lag_steps}"
            #temp_group[diff_lead_column_name] = temp_group[lead_column_name] - group[column]

        # Append the processed group to the lagged_df
        lagged_df = pd.concat([lagged_df, temp_group], axis=0)

    # Reset the index of the resulting DataFrame
    lagged_df = lagged_df.reset_index(drop=True)

    return lagged_df


