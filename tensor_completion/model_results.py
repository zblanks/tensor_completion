from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch

from cpd import CPD, create_feature_alphabet, DataModule
from nn import AutoRegressive

KATL_NUM_CONFIG = 27
NUM_CLUSTERS = 5


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_test_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", choices={"cpd", "nn"})
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--rank", type=int, default=10)
    return parser.parse_args()


def format_metrics(metrics: Dict[int, Dict[str, torch.Tensor]]) -> pd.DataFrame:
    log_loss = []
    ce = []
    acc = []

    for key in metrics.keys():
        log_loss.append(metrics[key]["test_log_loss"].item())
        ce.append(metrics[key]["test_ce"].item())
        acc.append(metrics[key]["test_acc"].item())

    data = {"log_loss": log_loss, "ce": ce, "acc": acc}
    return pd.DataFrame(data)


def main():
    args = parse_args()
    model_type: str = args.model_type
    seq_len: int = args.seq_len
    rank: int = args.rank

    p = Path("../data")

    if model_type == "cpd":
        feature_alphabet = create_feature_alphabet(seq_len)
        model = CPD(feature_alphabet, KATL_NUM_CONFIG, rank)
        dm = DataModule(p, seed=17)

        model_paths = []
        for seed in range(1, 21):
            model_paths.append(
                f"../models/rank={rank}_seqlen={seq_len}_seed={seed}.ckpt"
            )
    elif model_type == "nn":
        model = AutoRegressive(KATL_NUM_CONFIG)
        dm = DataModule(p, 17)

        model_paths = []
        for seed in range(1, 21):
            model_paths.append(f"../models/katl_nn_seed={seed}.ckpt")

    metrics = {}
    for (i, path) in enumerate(model_paths):
        model.load_state_dict(torch.load(path)["state_dict"])
        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(callbacks=[metrics_callback])
        trainer.test(model, datamodule=dm)
        metrics[i] = metrics_callback.metrics[0]

    df = format_metrics(metrics)
    df.to_csv(f"../data/{model_type}_oos_results.csv", index=False)


if __name__ == "__main__":
    main()
