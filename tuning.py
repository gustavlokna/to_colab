from pathlib import Path
import pandas as pd
from src.dataset import Dataset
from src.splitting import SplitterMonth, SplitterSummer
from sklearn.ensemble import GradientBoostingRegressor
import argparse

folder = Path("data_selected")

data = Dataset(folder, splitter=SplitterSummer())

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import pickle
from pathlib import Path

def hyperparameter_tuning(est, param_grid, data, splitter):

  
    if not splitter is None:
        data.splitter = splitter

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(
        estimator=est, param_grid=param_grid, scoring=scorer,
        cv = data.split_index(), 
        verbose=2, n_jobs=-1
    )
    
    # Fir the grid search
    X, y = data.X, data.y
    w = get_weights(y)

    grid_search.fit(X.values, y.pv_measurement)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params} - {grid_search.best_score_}")
    
    return best_params, grid_search

def increase_data(X, y):

    all_X, all_y = [X], [y]
    for factor, rel_months in [(5, [4, 7, 8]), (10, [5, 6])]:
        mask_X = X.index.map(lambda x: x.month in rel_months)
        mask_y = y.index.map(lambda x: x.month in rel_months)

        all_X.extend([X[mask_X]] * factor)
        all_y.extend([y[mask_y]] * factor)
    
    X = pd.concat(all_X)
    y = pd.concat(all_y)

    return X, y

def get_weights(y):

    def w(x, low=5, high=7, b = 0.5):

        m = x.month

        if m > high:
            return (m - high + b) ** 2
        
        elif m < low:
            return (low - m + b) ** 2
        
        else:
            return 1

    return y.index.map(w).values
    

def run_hp_search(model, grid, data, folder : Path, splitter = None):
    
    folder.mkdir(exist_ok=True, parents=True)
    
    for key, x in data.train_data.items():
        
        if key != "a":
            continue
        
        for i, split in enumerate(x.split()):
            _, grid_search = hyperparameter_tuning(model, grid, split["train"], splitter)

            data_val = split["val"]
            X_val, y_val = data_val.X.values, data_val.y.pv_measurement

            mae = mean_absolute_error(
                grid_search.best_estimator_.predict(X_val),
                y_val
            )
            print(mae)
            
            with open(folder / f"{key}_{i}.pkl", "wb+") as f:
                pickle.dump({"mae": mae, "grid_search": grid_search}, f)



if __name__ == "__main__":

    # # Parse inputs
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_estimators", type=float, nargs="+", default=[500, 1000], help="The epsilon values to use")
    # parser.add_argument("--max_depth", type=float, nargs="+", default=[5], help="The epsilon values to use")
    # parser.add_argument("--learning_rate", type=float, nargs="+", default=[0.05], help="The epsilon values to use")
    # parser = parser.parse_args()

    # # Grid 1 - Score 234.1
    # grid_gbr = {
    #     "n_estimators": parser.n_estimators, 
    #     "loss": ["absolute_error"],
    #     "max_depth": parser.max_depth,
    #     "learning_rate": parser.learning_rate,
    # }

    # gbr = GradientBoostingRegressor()

    # run_hp_search(gbr, grid_gbr, data, Path(f"_gbr_1"))

    import xgboost as xgb

    grid_XGB = {
        'learning_rate': [0.01],
        'max_depth': [10, 20],
        # 'subsample': [0.5, 0.7, 0.9],
        # 'colsample_bytree': [0.5, 0.7, 0.9],
        'n_estimators': [500, 1000],
        'objective': ['reg:squarederror']
    }

    xg_reg = xgb.XGBRegressor() 

    run_hp_search(xg_reg, grid_XGB, data, Path("grid_XGB"))
