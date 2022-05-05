from argparse import ArgumentParser
import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

KATL_NUM_CONFIG = 27
NUM_CLUSTERS = 5


def mean_log_loss(yhat: Tensor, y: Tensor, nconfig: int) -> Tensor:
    yhat = F.softmax(yhat, dim=1)  # Valid distribution along all configs
    y = F.one_hot(y, num_classes=nconfig)
    yhat = yhat.permute(0, 2, 1)  # R^(n x L X C)
    out = y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat)
    return -out.mean()


def accuracy(yhat: Tensor, y: Tensor) -> Tensor:
    yhat = F.softmax(yhat, dim=1)
    yhat = torch.argmax(yhat, dim=1)  # R^(n x L)
    return (yhat == y).sum() / (y.shape[0] * y.shape[1])


class Data(Dataset):
    def __init__(self, X: Tensor, Y: Tensor) -> None:
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.X[idx, :], self.Y[idx, :]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        p: Path,
        seed: int,
        seq_len: int = 4,
        nsteps: int = 12,
        val_size: float = 0.20,
        test_size: float = 0.05,
        batch_size: int = 64,
    ):
        super().__init__()
        self.p = p
        self.seed = seed
        self.seq_len = seq_len
        self.nsteps = nsteps
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size

    def get_count_data(self, count_type: str) -> pd.DataFrame:
        p = self.p / f"katl_{count_type}_labels.csv"
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
        p = self.p / "katl_airport_config.csv.bz2"
        df = pd.read_csv(p, parse_dates=["timestamp"])

        with open(self.p / "katl_label_map.pickle", "rb") as f:
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
        self.X = torch.from_numpy(df.to_numpy())
        self.Y = torch.from_numpy(labels.to_numpy())

    def setup(self, stage: Optional[str] = None) -> None:
        idx = torch.arange(self.X.shape[0])
        train = math.floor(idx.shape[0] * (1 - (self.val_size + self.test_size)))
        val = math.floor(idx.shape[0] * self.val_size)

        train_idx = idx[:train]
        val_idx = idx[train : train + val]
        test_idx = idx[train + val :]

        if stage == "fit" or stage is None:
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

        if stage == "test" or stage is None:
            self.X_test = self.X[test_idx, :]
            self.Y_test = self.Y[test_idx, :]

    def train_dataloader(self):
        train_loader = Data(self.X_train, self.Y_train)
        return DataLoader(
            train_loader, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        val_loader = Data(self.X_val, self.Y_val)
        return DataLoader(
            val_loader, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        test_loader = Data(self.X_test, self.Y_test)
        return DataLoader(
            test_loader, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class CPD(pl.LightningModule):
    def __init__(
        self,
        feature_alphabet: Dict[str, int],
        nconfig: int,
        rank: int,
        lr: float = 1e-3,
        nsteps: int = 12,
    ):

        super().__init__()
        self.rank = rank
        self.nconfig = nconfig
        self.lr = lr
        self.nsteps = nsteps

        # Initialize factor matrices with U ~ (0, 1) of appropriate size
        self.factors = nn.ParameterDict()
        for (feature, nidx) in feature_alphabet.items():
            self.factors[feature] = nn.Parameter(torch.rand(nidx, rank))

        # Initialize the factor for configurations
        self.factors["b"] = nn.Parameter(torch.rand(nconfig, rank))
        self.factors["c"] = nn.Parameter(torch.rand(nsteps, rank))

    def forward(self, x):
        # N = number of samples
        # C = number of configurations
        # L = number of lookahead timesteps
        # F = rank
        n = x.shape[0]
        X = torch.ones((n, self.rank), requires_grad=True)

        for i in range(x.shape[1]):
            X = X * torch.sigmoid(self.factors[str(i)][x[:, i], :])  # R^(n x F)

        X = X.unsqueeze(1)  # R^(n x 1 x F)
        B = self.factors["b"].repeat(n, 1, 1)  # R^(n x C x F)
        X = (X * B).unsqueeze(2)  # R^(n x C x 1 x F)

        C = self.factors["c"].unsqueeze(0).unsqueeze(0)  # R^(1 x 1 x L x F)
        X = X * C  # R^(n x C x L x F)
        return X.sum(dim=-1)  # R^(n x C x L)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        avg_log_loss = mean_log_loss(yhat, y, self.nconfig)
        self.log("train_loss", avg_log_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        val_loss = mean_log_loss(yhat, y, self.nconfig)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        test_log_loss = mean_log_loss(yhat, y, self.nconfig)
        test_ce = F.cross_entropy(yhat, y)
        test_acc = accuracy(yhat, y)
        self.log("test_log_loss", test_log_loss)
        self.log("test_ce", test_ce)
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_feature_alphabet(seq_len: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in range(seq_len):
        out[str(i)] = KATL_NUM_CONFIG

    # By construction, there are 12 lookahead samples for arrivals &
    # departures
    for i in range(seq_len, seq_len + 24):
        out[str(i)] = NUM_CLUSTERS

    return out


def main():
    args = parse_args()
    rank: int = args.rank
    seq_len: int = args.seq_len
    seed: int = args.seed

    p = Path("../data")
    pl.seed_everything(17)
    dm = DataModule(p, seed, seq_len=seq_len)

    feature_alphabet = create_feature_alphabet(seq_len)
    cpd = CPD(feature_alphabet, KATL_NUM_CONFIG, rank=rank)

    model_path = Path("../models")
    filename = f"rank={rank}_seqlen={seq_len}_seed={seed}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename=filename,
        monitor="val_loss",
        save_top_k=1,
    )

    early_stopping = EarlyStopping("val_loss", patience=5)
    callbacks = [early_stopping, checkpoint_callback]

    logger = TensorBoardLogger(
        "../logs",
        name=f"rank={rank}_seqlen={seq_len}",
        version=f"seed={seed}",
    )

    trainer = pl.Trainer(logger=logger, callbacks=callbacks)
    trainer.fit(cpd, datamodule=dm)


if __name__ == "__main__":
    main()
