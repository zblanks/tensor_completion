import math
from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch

from cpd import mean_log_loss

KATL_NUM_CONFIG = 27


class Data:
    def __init__(
        self,
        seed: int,
        data_dir: Path = Path("../data"),
        seq_len: int = 4,
        val_size: float = 0.2,
        test_size: float = 0.05,
        nsteps: int = 12,
    ):
        self.seed = seed
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.val_size = val_size
        self.test_size = test_size
        self.nsteps = nsteps

    def get_count_data(self, count_type: str) -> pd.DataFrame:
        p = self.data_dir / f"katl_{count_type}_labels.csv"
        df = pd.read_csv(p, parse_dates=["timestamp", "forecast_timestamp"])

        n = df.shape[0] // 16
        timesteps = torch.tile(torch.arange(16), (n,))
        df["forecast_timestamp"] = timesteps
        df = df.pivot(
            index="timestamp",
            columns="forecast_timestamp",
            values=f"estimated_{count_type}",
        )

        df = df.loc[:, torch.arange(self.nsteps)]
        return df

    def get_configs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p = self.data_dir / "katl_airport_config.csv.bz2"
        df = pd.read_csv(p, parse_dates=["timestamp"])

        with open(self.data_dir / "katl_label_map.pickle", "rb") as f:
            label_map = pickle.load(f)

        df = df.set_index("timestamp")
        df = df.replace(label_map)
        df = df.resample("30min").ffill().dropna()
        df = df.astype("category")
        s = pd.Series(df.airport_config, index=df.index)
        configs = s.cat.categories
        config_map = dict(zip(configs, range(len(configs))))
        s = s.replace(config_map)

        n = s.shape[0] - self.seq_len - self.nsteps + 1
        idx = torch.tile(torch.arange(self.seq_len), (n, 1))
        idx += torch.arange(n).unsqueeze(1)
        ts = s.index[idx[:, -1].numpy()]
        arr = torch.from_numpy(s.to_numpy())
        configs = pd.DataFrame(arr[idx], index=ts)

        idx = torch.tile(torch.arange(self.seq_len, self.seq_len + self.nsteps), (n, 1))
        idx += torch.arange(n).unsqueeze(1)
        labels = pd.DataFrame(arr[idx], index=ts)

        # Need a common starting point that avoids issue of different sequence length
        start_time = pd.Timestamp("2020-11-06 13:00:00")
        configs = configs.loc[configs.index >= start_time, :]
        labels = labels.loc[labels.index >= start_time, :]
        return configs, labels

    def prepare_data(self):
        arrival = self.get_count_data("arrival")
        departure = self.get_count_data("departure")
        configs, labels = self.get_configs()

        # Need to get the intersection of timestamps to ensure we're
        # working with the same data
        idx = configs.index.intersection(arrival.index)
        arrival = arrival.loc[idx, :]
        departure = departure.loc[idx, :]
        configs = configs.loc[idx, :]
        labels = labels.loc[idx, :]

        # Merge configs, arrival, and departure into single feature matrix
        df = pd.concat([configs, arrival, departure], axis=1)
        self.X = df.to_numpy()
        self.Y = labels.to_numpy()

    def setup(self) -> None:
        idx = torch.arange(self.X.shape[0])
        train = math.floor(idx.shape[0] * (1 - (self.val_size + self.test_size)))
        val = math.floor(idx.shape[0] * self.val_size)

        train_idx = idx[:train]
        val_idx = idx[train : train + val]
        test_idx = idx[train + val :]

        self.X_train = self.X[train_idx, :]
        self.Y_train = self.Y[train_idx, :]

        # Bootstrap re-sampling of the training set to allow for
        # statistical comparisons between models
        g = torch.Generator().manual_seed(self.seed)
        n = self.X_train.shape[0]
        boot = torch.randint(0, n, (n,), generator=g)
        self.X_train = self.X_train[boot, :]
        self.Y_train = self.Y_train[boot, :]

        self.X_val = self.X[val_idx, :]
        self.Y_val = self.Y[val_idx, :]

        self.X_test = self.X[test_idx, :]
        self.Y_test = self.Y[test_idx, :]

    def get_data(self) -> None:
        self.prepare_data()
        self.setup()


def update_y(y: np.ndarray) -> np.ndarray:
    # We don't have any instances of label 25 in the training set, so
    # so the system thinks we only have 26 classes; we'll have to manually
    # scale label 26 -> 25 to correct this issue
    idx = np.where(y == 26)
    y[idx] -= 1
    return y


def get_score(seed: int) -> float:
    data = Data(seed=seed)
    data.get_data()

    rf = RandomForestClassifier(random_state=17, n_jobs=-1)
    rf.fit(data.X_train, data.Y_train)

    n = data.X_test.shape[0]
    yhat = rf.predict_proba(data.X_test)
    nsteps = len(yhat)
    nconfig = yhat[0].shape[1]
    yhat = np.concatenate(yhat).reshape(n, nconfig, nsteps)
    yhat = torch.from_numpy(yhat)
    y = torch.from_numpy(update_y(data.Y_test))
    return mean_log_loss(yhat, y, nconfig).item()


def bootstrap(s: pd.Series, bootstrap_samples: int = 1000) -> pd.Series:
    g = torch.Generator().manual_seed(17)
    n = s.shape[0]
    idx = torch.randint(0, n, size=(bootstrap_samples, n), generator=g)
    arr = torch.from_numpy(s.to_numpy())  # (n,)
    arr = arr[idx]  # (B, n)
    arr = arr.mean(dim=-1)  # (B,)
    return pd.Series(arr.numpy(), name="log_loss")


def main():
    scores = []
    for seed in range(1, 21):
        scores.append(get_score(seed))

    s = pd.Series(scores, name="log_loss")
    s.to_csv("../data/rf_oos_results.csv", index=False)


if __name__ == "__main__":
    main()
