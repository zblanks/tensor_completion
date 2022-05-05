from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(x: np.ndarray, nclusters: int) -> KMeans:
    kmeans = KMeans(n_clusters=nclusters, random_state=17)
    kmeans.fit(x)
    return kmeans


def update_labels(kmeans: KMeans) -> np.ndarray:
    # The cluster labels are arbitrary, but since we're working only in a 1D
    # domain, almost surely, they will partition the space between [0, 1], so
    # we can smoothly re-map their values where 0 => lowest, ...
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()  # (n,); n = number of clusters
    label_map = dict(zip(np.argsort(centers), np.arange(len(centers))))
    labels = np.vectorize(label_map.get)(labels)
    return labels


def save_cluster_labels(
    path: str, df: pd.DataFrame, count_type: str, kmeans: KMeans
) -> None:
    labels = update_labels(kmeans)
    df[f"estimated_{count_type}"] = labels
    df.to_csv(path)


def save_cluster_results(path: str, nclusters: int, kmeans: KMeans) -> None:
    data = {"nclusters": [nclusters], "loss": [kmeans.inertia_]}
    df = pd.DataFrame(data)
    df.to_csv(path, mode="a", index=False, header=False)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--nclusters", type=int)
    parser.add_argument("--count_type", choices={"arrival", "departure"})
    parser.add_argument("--save_labels", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nclusters: int = args.nclusters
    count_type: str = args.count_type
    save_results: bool = args.save_results
    save_labels: bool = args.save_labels

    p = f"../data/katl_estimated_{count_type}.csv.bz2"
    df = pd.read_csv(p, parse_dates=["timestamp", "forecast_timestamp"])
    df = df.set_index(["timestamp", "forecast_timestamp"])
    x = df.to_numpy()

    kmeans = run_kmeans(x, nclusters)

    if save_results:
        path = f"../data/katl_{count_type}_loss.csv"
        save_cluster_results(path, nclusters, kmeans)

    if save_labels:
        path = f"../data/katl_{count_type}_labels.csv"
        save_cluster_labels(path, df, count_type, kmeans)


if __name__ == "__main__":
    main()
