import numpy as np
from sklearn.model_selection import KFold

from src.dataset import Data

class SplitterMonth:

    def __init__(self, split_str : str = "%d-%m-%y", *args, **kwargs) -> None:
        self.kf = KFold(*args, **kwargs)
        self.split_str = split_str

    def __call__(self, data : Data, return_index = False):

        # Add new key
        splits = []

        # Compute valid months
        times_X = data.X.index.map(lambda x: str(x.strftime(self.split_str)))
        times_y = data.y.index.map(lambda x: str(x.strftime(self.split_str)))

        # Only split on single months
        month_set = list(set(times_X))
        idx_to_split = np.array(month_set).reshape((-1, 1))

        # Split using KF-CV
        for train_indicies, _ in self.kf.split(idx_to_split):
            
            # Compute if X and y values are in the correct month
            train_months = [month_set[idx] for idx in train_indicies]
            idx_X = times_X.isin(train_months)
            idx_y = times_y.isin(train_months)
          
            X_train = data.X[idx_X]
            y_train = data.y[idx_y]
            
            X_val = data.X[~idx_X]
            y_val = data.y[~idx_y]

            if return_index:
                splits.append((np.where(idx_X)[0], np.where(~idx_X)[0]))

            else:
                splits.append({
                    "train": Data(X_train, y_train, self),
                    "val": Data(X_val, y_val, self)
                })

        return splits


class SplitterSummer:

    def __init__(self, summer_months = [5, 6], *args, **kwargs) -> None:
        self.kf = KFold(*args, **kwargs)
        self.summer_months = summer_months

    def __call__(self, data : Data, return_index = False):

        # Add new key
        splits = []

        # Compute valid months
        years = set(data.X.index.map(lambda x: x.year if x.month in self.summer_months else -1).values)

        years = [y for y in years if y > 0]

        for year in years:

            if years == -1:
                continue

            mask = data.X.index.map(lambda x: x.year == year and x.month in self.summer_months).values
            mask_y = data.y.index.map(lambda x: x.year == year and x.month in self.summer_months).values

            # Test that X and y equal
            assert np.any(mask & ~mask_y) == False
          
            X_train = data.X[~mask]
            y_train = data.y[~mask]
            
            X_val = data.X[mask]
            y_val = data.y[mask]

            if return_index:
                splits.append((np.where(~mask)[0], np.where(mask)[0]))

            else:
                splits.append({
                    "train": Data(X_train, y_train, self),
                    "val": Data(X_val, y_val, self)
                })

        return splits