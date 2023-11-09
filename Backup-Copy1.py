{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "c29fae58",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Already up to date.\n"
     ]
    }
   ],
   "source": [
    "!git pull"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ede81172",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "rolling_aggregates_list = ['air_density_2m:kgm3','dew_point_2m:K','effective_cloud_cover:p','relative_humidity_1000hPa:p','sfc_pressure:hPa','sun_elevation:d','t_1000hPa:K','total_cloud_cover:p']\n",
    "\n",
    "selected_features = ['date_forecast',\n",
    " 'absolute_humidity_2m:gm3',\n",
    " 'air_density_2m:kgm3',\n",
    " 'ceiling_height_agl:m',\n",
    " 'clear_sky_energy_1h:J',\n",
    " 'clear_sky_rad:W',\n",
    " 'cloud_base_agl:m',\n",
    " #'dew_or_rime:idx',\n",
    " 'dew_point_2m:K',\n",
    " 'diffuse_rad:W',\n",
    " 'diffuse_rad_1h:J',\n",
    " 'direct_rad:W',\n",
    " 'direct_rad_1h:J',\n",
    " 'effective_cloud_cover:p',\n",
    " #'elevation:m',\n",
    " #'fresh_snow_12h:cm',\n",
    " #'fresh_snow_1h:cm',\n",
    " 'fresh_snow_24h:cm',\n",
    " #'fresh_snow_3h:cm',\n",
    " #'fresh_snow_6h:cm',\n",
    " #'is_day:idx',\n",
    " #'is_in_shadow:idx',\n",
    " 'msl_pressure:hPa',\n",
    " 'precip_5min:mm',\n",
    " 'precip_type_5min:idx',\n",
    " 'pressure_100m:hPa',\n",
    " #'pressure_50m:hPa',\n",
    " #'prob_rime:p',\n",
    " #'rain_water:kgm2',\n",
    " 'relative_humidity_1000hPa:p',\n",
    " 'sfc_pressure:hPa',\n",
    " 'snow_density:kgm3',\n",
    " #'snow_depth:cm',\n",
    " #'snow_drift:idx',\n",
    " #'snow_melt_10min:mm',\n",
    " 'snow_water:kgm2',\n",
    " 'sun_azimuth:d',\n",
    " 'sun_elevation:d',\n",
    " 'super_cooled_liquid_water:kgm2',\n",
    " 't_1000hPa:K',\n",
    " 'total_cloud_cover:p',\n",
    " 'visibility:m',\n",
    " 'wind_speed_10m:ms',\n",
    " 'wind_speed_u_10m:ms',\n",
    " 'wind_speed_v_10m:ms',\n",
    " #'wind_speed_w_1000hPa:ms',\n",
    " 'dif_dat_rad',\n",
    " 'hour',\n",
    " #'minute',\n",
    " 'month',\n",
    " #'time_decimal',\n",
    " 'hour_sin',\n",
    " #'hour_cos',\n",
    " \"direct_plus_diffuse\",\n",
    " \"direct_plus_diffuse_1h\"\n",
    " ]\n",
    "\n",
    "selected_features.remove('ceiling_height_agl:m')\n",
    "selected_features.remove('cloud_base_agl:m')\n",
    "selected_features.remove('snow_density:kgm3')\n",
    "wanted_months = [3,4,5,6,7,8,9]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "7bb40676",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from src.functions import * \n",
    "import operator \n",
    "def preprocess(data):\n",
    "    \n",
    "    out = {}\n",
    "    for key, values in data.items():\n",
    "        \n",
    "        \n",
    "        \n",
    "        # Get the local variables\n",
    "        X_est, X_obs, X_test, y = (\n",
    "            values[key] for key in [\n",
    "                \"X_train_estimated\", \"X_train_observed\", \n",
    "                \"X_test_estimated\", \"train_targets\"\n",
    "            ]\n",
    "        )\n",
    "        \n",
    "        # Add y imputation\n",
    "        if key == \"b\":\n",
    "            y, _ = augment_y_b(y)\n",
    "        elif key == \"c\":\n",
    "            y, _ = augment_y_c(y)\n",
    "\n",
    "        # Preprocess X\n",
    "        X_est, X_obs, X_test = (\n",
    "            df_feature_selection(\n",
    "                add_all_features(mean_df(X)), selected_features\n",
    "            )\n",
    "            for X in [X_est, X_obs, X_test]\n",
    "        )\n",
    "        \n",
    "        \n",
    "        # Only select subset of observed train data\n",
    "        X_obs = subset_months(X_obs, wanted_months)\n",
    "        \n",
    "        # Concat and align\n",
    "        X_train = pd.concat([X_obs, X_est], axis = 0)\n",
    "        X_train =    add_lag_and_lead_features(X_train, 1, [\"direct_plus_diffuse\",\"direct_plus_diffuse_1h\"])\n",
    "        X_train =    rolling_aggregates(X_train.copy(), rolling_aggregates_list, 24)\n",
    "        X_train =    create_new_feature(X_train.copy(), 'total_cloud_cover:p', 'direct_rad:W', '/')\n",
    "        X_train, y = resize_training_data(X_train, y)\n",
    "\n",
    "        out[key] = {\n",
    "            \"X_train\": X_train,\n",
    "            \"y_train\": y,\n",
    "            \"X_test\": X_test\n",
    "        }\n",
    "    \n",
    "    return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36d72856-87d5-4033-a55f-d186cd919235",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "6f6c7b6f",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n",
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n",
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n",
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n",
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n",
      "C:\\Users\\Gustav\\Documents\\maskinlaering\\ML_prosject\\Johan_Mvp\\Johan_mvp\\src\\functions.py:141: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.\n",
      "  averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()\n"
     ]
    }
   ],
   "source": [
    "#!git pull \n",
    "from src.dataset import Dataset\n",
    "from src.splitting import SplitterMonth\n",
    "\n",
    "data = Dataset(splitter=SplitterMonth(n_splits=5, shuffle=False), preprocess=preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "e47c3bcf",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import make_scorer, mean_absolute_error\n",
    "\n",
    "def hyperparameter_tuning(est, param_grid, data):\n",
    "\n",
    "    # Create the scorer\n",
    "    scorer = make_scorer(mean_absolute_error, greater_is_better=False)\n",
    "\n",
    "    # Create the grid search\n",
    "    grid_search = GridSearchCV(\n",
    "        estimator=est, param_grid=param_grid, scoring=scorer,\n",
    "        cv = 5, # data.split_index(), \n",
    "        verbose=2, n_jobs=-1\n",
    "    )\n",
    "\n",
    "    # Should be done in pipeline\n",
    "    X = data.X #.dropna(axis=1)\n",
    "    X = X[[c for c in X.columns if not c in ['date_calc', 'date_forecast']]]\n",
    "    \n",
    "    # Fir the grid search\n",
    "    grid_search.fit(X, data.y[\"pv_measurement\"])\n",
    "\n",
    "    # Get the best parameters\n",
    "    best_params = grid_search.best_params_\n",
    "    print(f\"Best parameters b: {best_params} - {grid_search.best_score_}\")\n",
    "    \n",
    "    return best_params, grid_search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "403f43b8-8b38-432e-91b4-5178c707c9b7",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'grid_search' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[54], line 6\u001b[0m\n\u001b[0;32m      3\u001b[0m data_val \u001b[38;5;241m=\u001b[39m data\u001b[38;5;241m.\u001b[39mtrain_data[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124ma\u001b[39m\u001b[38;5;124m\"\u001b[39m]\u001b[38;5;241m.\u001b[39msplit()[\u001b[38;5;241m0\u001b[39m][\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mval\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m      4\u001b[0m X_val, y_val \u001b[38;5;241m=\u001b[39m data_val\u001b[38;5;241m.\u001b[39mX, data_val\u001b[38;5;241m.\u001b[39my\n\u001b[1;32m----> 6\u001b[0m X_val \u001b[38;5;241m=\u001b[39m X_val[grid_search\u001b[38;5;241m.\u001b[39mfeature_names_in_]\n\u001b[0;32m      8\u001b[0m mean_absolute_error(\n\u001b[0;32m      9\u001b[0m     grid_search\u001b[38;5;241m.\u001b[39mbest_estimator_\u001b[38;5;241m.\u001b[39mpredict(X_val),\n\u001b[0;32m     10\u001b[0m     y_val\u001b[38;5;241m.\u001b[39mpv_measurement\n\u001b[0;32m     11\u001b[0m )\n",
      "\u001b[1;31mNameError\u001b[0m: name 'grid_search' is not defined"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_absolute_error\n",
    "\n",
    "data_val = data.train_data[\"a\"].split()[0][\"val\"]\n",
    "X_val, y_val = data_val.X, data_val.y\n",
    "\n",
    "X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "mean_absolute_error(\n",
    "    grid_search.best_estimator_.predict(X_val),\n",
    "    y_val.pv_measurement\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4f7a24e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "\n",
    "grid_GBR = {\n",
    "    \"n_estimators\": [100, 200, 500], \n",
    "    \"loss\": [\"absolute_error\"],\n",
    "    \"max_depth\": [None],\n",
    "    \"learning_rate\": [0.01, 0.1, 1],\n",
    "}\n",
    "\n",
    "GBR = GradientBoostingRegressor()\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(GBR, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ...\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "33a27467-124d-446b-ac6a-c4e0cb46a958",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from catboost import CatBoostRegressor\n",
    "\n",
    "grid_CB = {\n",
    "        \"iterations\": [1000],\n",
    "        \"learning_rate\": [1e-3, 0.1],\n",
    "        \"depth\": [1, 10],\n",
    "        \"subsample\": [0.05, 1.0],\n",
    "        \"colsample_bylevel\": [ 0.05, 1.0],\n",
    "        \"min_data_in_leaf\": [ 1, 100],\n",
    "    }\n",
    "\n",
    "CB = CatBoostRegressor()\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(CB, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77583aff-be14-4404-b919-92c9649235b0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 32 candidates, totalling 160 fits\n"
     ]
    }
   ],
   "source": [
    "_, grid_search = hyperparameter_tuning(CB, params, data.train_data[\"a\"].split()[0][\"train\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e4a9b09f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "grid_RF = {\n",
    "    \"criterion\": [\"absolute_error\"],\n",
    "    \"n_estimators\": [200], \n",
    "    \"max_depth\": [None],\n",
    "    \"min_samples_leaf\": [4]\n",
    "}\n",
    "\n",
    "rf = RandomForestRegressor()\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(CB, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "606b9225-7d80-4f87-8151-23b70b12586c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "\n",
    "grid_KNN = {\n",
    "    'n_neighbors': [3, 5, 7, 10],\n",
    "    'weights': ['uniform', 'distance'],\n",
    "    \"algorithm\" : [\"auto\", \"ball_tree\", \"kd_tree\", \"brute\"]\n",
    "    \"leaf_size\" : [10, 30, 50]\n",
    "    'metric': ['euclidean', 'manhattan']\n",
    "}\n",
    "\n",
    "KNN = KNeighborsRegressor()\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(CB, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68bdc0e3-356f-453c-9bb2-9ba81b77a080",
   "metadata": {},
   "outputs": [],
   "source": [
    "from lightgbm import LGBMRegressor\n",
    "\n",
    "grid_LightGBM = {\n",
    "    'colsample_bytree': (0.3, 1.0),    # Range: 0.3 to 1.0\n",
    "    'learning_rate': (0.001, 0.3),     # Range: 0.001 to 0.3\n",
    "    'max_depth': (-1, 15),             # Range: -1 (no limit) to 15\n",
    "    'n_estimators': (50, 1000),        # Range: 50 to 1000\n",
    "    'objective': 'regression',         # Typically fixed, based on task\n",
    "    'subsample': (0.5, 1.0),           # Range: 0.5 to 1.0\n",
    "    'min_data_in_leaf': (1, 100),      # Range: 1 to 100\n",
    "    'num_leaves': (20, 100)            # Range: 20 to 100\n",
    "}\n",
    "\n",
    "LGBM = LGBMRegressor()\n",
    "\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(CB, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bfc3d12b-dd92-4e5c-bc7b-e402e25bafe0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import xgboost as xgb \n",
    "\n",
    "grid_XGB = {\n",
    "    'learning_rate': [0.01, 0.1, 0.2],\n",
    "    'max_depth': [3, 4, 5],\n",
    "    'subsample': [0.5, 0.7, 0.9],\n",
    "    'colsample_bytree': [0.5, 0.7, 0.9],\n",
    "    'n_estimators': [100, 200, 500],\n",
    "    'objective': ['reg:squarederror']\n",
    "}\n",
    "\n",
    "xg_reg = xgb.XGBRegressor() \n",
    "\n",
    "\n",
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(CB, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "0b5f50c4",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#_, grid_search = hyperparameter_tuning(rf, grid, data.train_data[\"a\"].split()[0][\"train\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "ae214260",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#grid_search.cv_results_, grid_search.feature_names_in_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "2c098e71",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "256.6385384503628"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_absolute_error\n",
    "\n",
    "data_val = data.train_data[\"a\"].split()[0][\"val\"]\n",
    "X_val, y_val = data_val.X, data_val.y\n",
    "\n",
    "X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "mean_absolute_error(\n",
    "    grid_search.best_estimator_.predict(X_val),\n",
    "    y_val.pv_measurement\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "580c2fef",
   "metadata": {},
   "outputs": [],
   "source": [
    "for key, x in data.train_data:\n",
    "    for split in x.split():\n",
    "        _, grid_search = hyperparameter_tuning(rf, grid, split[\"train\"])\n",
    "        \n",
    "        data_val = split[\"val\"]\n",
    "        X_val, y_val = data_val.X, data_val.y\n",
    "        X_val = X_val[grid_search.feature_names_in_]\n",
    "\n",
    "        mae = mean_absolute_error(\n",
    "            grid_search.best_estimator_.predict(X_val),\n",
    "            y_val.pv_measurement\n",
    "        )\n",
    "        print(mae) \n",
    "        # ...\n",
    "        "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
