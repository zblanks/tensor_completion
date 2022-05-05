from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd


def make_plot(df: pd.DataFrame, count_type: str) -> None:
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df.nclusters, df.loss, marker="o", color="black")
    ax.set_xlabel("Number of Clusters", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(
        f"Expected {count_type.capitalize()}s Cluster Elbow Plot", fontsize=32, y=1.05
    )

    path = f"../figures/katl_{count_type}_elbow_plot.pdf"
    plt.savefig(path, transparent=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--count_type", choices={"arrival", "departure"})
    return parser.parse_args()


def main():
    args = parse_args()
    count_type: str = args.count_type

    p = f"../data/katl_{count_type}_loss.csv"
    df = pd.read_csv(p, header=None)
    df.columns = ["nclusters", "loss"]
    make_plot(df, count_type)


if __name__ == "__main__":
    main()
