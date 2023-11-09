import pandas as pd
from pathlib import Path
import pickle

class Data:
    """
    Single data with X and y
    """

    def __init__(self, X, y, splitter = None) -> None:
        self.X, self.y = X, y
        self.splitter = splitter

    def split(self):
        return self.splitter(self)

    def split_index(self):
        return self.splitter(self, return_index=True)


class Dataset:

    def __init__(self, data_dir : Path = Path("data"), splitter = None, preprocess=None) -> None:
        """

        Description of format for preprocess
            Preprocess the data - make it the right format

            Return {
              "a": {"X_train": ..., "y_train": ..., "X_test": ...},
              "b": {"X_train": ..., "y_train": ..., "X_test": ...},
            }
        """

        try:
            self.load(data_dir)
            print(f"Loaded pre-stored dataset from {data_dir}")

        except Exception:

            # Get all data
            self.data = {
                p.name.lower(): {
                    key.stem: pd.read_parquet(key)
                    for key in p.glob("*")
                } for p in data_dir.glob("*")
            }
            print(f"Loaded parquet dataset from {data_dir}")
        
        # Potentially preprocess
        if not preprocess is None:
            self.data = preprocess(self.data)
            print("Pre-processed the dataset")
        
        # Root dir
        self.splitter = splitter

    @property
    def X_test(self):
        """
        Return all test data
        """
        return {
            key: value["X_test"]
            for key, value in self.data.items()
        }

    @property
    def X_train(self):
        """
        Return all train data
        """
        return {
            key: value["X_train"]
            for key, value in self.data.items()
        }

    @property
    def y_train(self):
        """
        Return all train data
        """
        return {
            key: value["y_train"]
            for key, value in self.data.items()
        }

    @property
    def train_data(self):
        
        X, y = self.X_train, self.y_train

        return {
            key : Data(X[key], y[key], self.splitter) 
            for key in X.keys()
        }

    def save(self, data_dir : Path):

        # Make folder
        data_dir.mkdir(exist_ok=True, parents=True)

        # Save data
        with open(data_dir / "dataset.pkl", "wb+") as f:
            pickle.dump({"data": self.data, "splitter": self.splitter}, f)

    def load(self, data_dir : Path):

        # Load data
        with open(data_dir / "dataset.pkl", "rb") as f:
            for key, value in pickle.load(f).items():
                setattr(self, key, value)
