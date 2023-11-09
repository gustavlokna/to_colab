import pandas as pd
from pathlib import Path

def load_data(file):
    path : Path = Path(file)
    # Load train
    dfs_train = {
        p.name.lower(): (
            pd.read_parquet(p / "X_train_estimated.parquet"),
            pd.read_parquet(p / "X_train_observed.parquet"),
            pd.read_parquet(p / "train_targets.parquet")
        )
        for p in path.glob("*")
    }

    # Load test
    dfs_test = {
        p.name.lower(): pd.read_parquet(p / "X_test_estimated.parquet")
        for p in path.glob("*")
    }

    # Combine train and test
    return {
        "train": dfs_train,
        "test": dfs_test
    }

def save_data(df, path : Path):

    df_train = df["train"]

    for key, (X_e, X_o, y) in df_train.items():

        # Save all data
        folder = path / key
        folder.mkdir(parents=True, exist_ok=True)
        df["test"][key].to_parquet(folder / "X_test_estimated.parquet")
        X_e.to_parquet(folder / "X_train_estimated.parquet")
        X_o.to_parquet(folder / "X_train_observed.parquet")
        y.to_parquet(folder / "train_targets.parquet")

        
def save_training_data(df, path : Path):

    df_train = df["train"]

    for key, (X,_, y) in df_train.items():

        # Save all data
        folder = path / key
        folder.mkdir(parents=True, exist_ok=True)
        df["test"][key].to_parquet(folder / "X_test_estimated.parquet")
        X.to_parquet(folder / "X_train.parquet")
        y.to_parquet(folder / "train_targets.parquet")

def load_training_data(file):
    path : Path = Path(file)
    # Load train
    dfs_train = {
        p.name.lower(): (
            pd.read_parquet(p / "X_train.parquet"),
            pd.read_parquet(p / "train_targets.parquet"),
        )
        for p in path.glob("*")
    }

    # Load test
    dfs_test = {
        p.name.lower(): pd.read_parquet(p / "X_test_estimated.parquet")
        for p in path.glob("*")
    }

    # Combine train and test
    return {
        "train": dfs_train,
        "test": dfs_test
    }