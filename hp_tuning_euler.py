rolling_aggregates_list = ['air_density_2m:kgm3','dew_point_2m:K','effective_cloud_cover:p','relative_humidity_1000hPa:p','sfc_pressure:hPa','sun_elevation:d','t_1000hPa:K','total_cloud_cover:p']

selected_features = ['date_forecast',
 'absolute_humidity_2m:gm3',
 'air_density_2m:kgm3',
 'ceiling_height_agl:m',
 'clear_sky_energy_1h:J',
 'clear_sky_rad:W',
 'cloud_base_agl:m',
 #'dew_or_rime:idx',
 'dew_point_2m:K',
 'diffuse_rad:W',
 'diffuse_rad_1h:J',
 'direct_rad:W',
 'direct_rad_1h:J',
 'effective_cloud_cover:p',
 #'elevation:m',
 #'fresh_snow_12h:cm',
 #'fresh_snow_1h:cm',
 'fresh_snow_24h:cm',
 #'fresh_snow_3h:cm',
 #'fresh_snow_6h:cm',
 #'is_day:idx',
 #'is_in_shadow:idx',
 'msl_pressure:hPa',
 'precip_5min:mm',
 'precip_type_5min:idx',
 'pressure_100m:hPa',
 #'pressure_50m:hPa',
 #'prob_rime:p',
 #'rain_water:kgm2',
 'relative_humidity_1000hPa:p',
 'sfc_pressure:hPa',
 'snow_density:kgm3',
 #'snow_depth:cm',
 #'snow_drift:idx',
 #'snow_melt_10min:mm',
 'snow_water:kgm2',
 'sun_azimuth:d',
 'sun_elevation:d',
 'super_cooled_liquid_water:kgm2',
 't_1000hPa:K',
 'total_cloud_cover:p',
 'visibility:m',
 'wind_speed_10m:ms',
 'wind_speed_u_10m:ms',
 'wind_speed_v_10m:ms',
 #'wind_speed_w_1000hPa:ms',
 'dif_dat_rad',
 'hour',
 #'minute',
 'month',
 #'time_decimal',
 'hour_sin',
 #'hour_cos',
 "direct_plus_diffuse",
 "direct_plus_diffuse_1h"
 ]

selected_features.remove('ceiling_height_agl:m')
selected_features.remove('cloud_base_agl:m')
selected_features.remove('snow_density:kgm3')
wanted_months = [3,4,5,6,7,8,9]



from tabnanny import verbose
from src.functions import * 
import operator 
def preprocess(data):
    
    out = {}
    for key, values in data.items():
        
        
        
        # Get the local variables
        X_est, X_obs, X_test, y = (
            values[key] for key in [
                "X_train_estimated", "X_train_observed", 
                "X_test_estimated", "train_targets"
            ]
        )
        
        # Add y imputation
        if key == "b":
            y, _ = augment_y_b(y)
        elif key == "c":
            y, _ = augment_y_c(y)

        # Preprocess X
        X_est, X_obs, X_test = (
            df_feature_selection(
                add_all_features(mean_df(X)), selected_features
            )
            for X in [X_est, X_obs, X_test]
        )
        
        
        # Only select subset of observed train data
        X_obs = subset_months(X_obs, wanted_months)
        
        # Concat and align
        X_train = pd.concat([X_obs, X_est], axis = 0)
        X_train =    add_lag_and_lead_features(X_train, 1, ["direct_plus_diffuse","direct_plus_diffuse_1h"])
        # X_train =    rolling_aggregates(X_train.copy(), rolling_aggregates_list, 24)
        # X_train =    create_new_feature(X_train.copy(), 'total_cloud_cover:p', 'direct_rad:W', '/')
        X_train, y = resize_training_data(X_train, y)

        out[key] = {
            "X_train": X_train,
            "y_train": y,
            "X_test": X_test
        }
    
    return out



#!git pull 
from src.dataset import Dataset
from src.splitting import SplitterMonth

data = Dataset(splitter=SplitterMonth(n_splits=3, shuffle=False), preprocess=preprocess)



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import pickle
from pathlib import Path

def hyperparameter_tuning(est, param_grid, data):

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(
        estimator=est, param_grid=param_grid, scoring=scorer,
        cv = data.split_index(), 
        verbose=3, n_jobs=-1
    )

    # Should be done in pipeline
    
    X = data.X #.dropna(axis=1)    
    X = X[[c for c in X.columns if not c in ['date_calc', 'date_forecast']]]
    
    # Fir the grid search
    grid_search.fit(X, data.y["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params} - {grid_search.best_score_}")
    
    return best_params, grid_search


def run_hp_search(model, grid, data, folder : Path, splitter = None):
    
    folder.mkdir(exist_ok=True, parents=True)
    
    for key, x in data.train_data.items():
        
        
        if not splitter is None:
            x.splitter = splitter
    
    for key, x in data.train_data.items():
        for i, split in enumerate(x.split()):
            _, grid_search = hyperparameter_tuning(model, grid, split["train"])

            data_val = split["val"]
            X_val, y_val = data_val.X, data_val.y
            X_val = X_val[grid_search.feature_names_in_]

            mae = mean_absolute_error(
                grid_search.best_estimator_.predict(X_val),
                y_val.pv_measurement
            )
            print(key, i, mae)
            
            with open(folder / f"{key}_{i}.pkl", "wb+") as f:
                pickle.dump({"mae": mae, "grid_search": grid_search}, f)


grid_gbr = {
    "n_estimators": [200, 500, 1000], 
    "loss": ["absolute_error"],
    "max_depth": [None, 10 , 15],
    "learning_rate": [0.01, 0.05, 0.1],
}

grid_cb = {
        "iterations": [1000],
        "learning_rate": [1e-3, 0.05,  0.1],
        "depth": [8, 10],
        "subsample": [0.05, 1.0],
        "colsample_bylevel": [ 0.05, 1.0],
    }

grid_RF = {
    "criterion": ["absolute_error"],
    "n_estimators": [200, 500, 1000], 
    "max_depth": [None , 10, 16],
    "min_samples_leaf": [3, 4, 5]
}

grid_KNN = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size" : [10, 30, 50],
    'metric': ['euclidean', 'manhattan'],
}


grid_LightGBM = {
    'learning_rate': (0.01, 0.1),      # Range: 0.01 to 0.1
    'max_depth': (5, 12, 16),              # Range: 5 to 12
    'n_estimators': (200, 500, 1000),        # Range: 100 to 500
    'objective': 'regression',         # Typically fixed, based on task
}

grid_XGB = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 10],
    'n_estimators': [100, 200, 500],
    'objective': ['reg:squarederror']
}


# from sklearn.ensemble import GradientBoostingRegressor
# gbr = GradientBoostingRegressor()
# run_hp_search(gbr, grid_gbr, data, Path("gbr"), splitter = SplitterMonth(n_splits=5, shuffle=False))


# from catboost import CatBoostRegressor
# cb = CatBoostRegressor(verbose=0)
# run_hp_search(cb, grid_cb, data, Path("cb"), splitter = SplitterMonth(n_splits=5, shuffle=False))


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
run_hp_search(rf, grid_RF, data, Path("rf"), splitter = SplitterMonth(n_splits=5, shuffle=False))


from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor()
run_hp_search(KNN, grid_KNN, data, Path("KNN"), splitter = SplitterMonth(n_splits=5, shuffle=False))


import xgboost as xgb 
xg_reg = xgb.XGBRegressor() 
run_hp_search(xg_reg, grid_XGB, data, Path("grid_XGB"), splitter = SplitterMonth(n_splits=5, shuffle=False))


from lightgbm import LGBMRegressor
LGBM = LGBMRegressor()
run_hp_search(LGBM, grid_LightGBM, data, Path("LGBM"), splitter = SplitterMonth(n_splits=5, shuffle=False))
