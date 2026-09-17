"""t-SNE / UMAP visualization of pooled clip-level embeddings (train+val only;
test labels are hidden). Sweeps a couple of perplexity/n_neighbors settings and
saves all of them -- report which was most stable, not cherry-picked for looks.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASETS, CACHE_DIR, RESULTS_DIR
from src.train_probe import load_cached


def plot_embedding(coords, labels, label_names, title, out_path):
    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(label_names):
        mask = labels == i
        plt.scatter(coords[mask, 0], coords[mask, 1], s=12, color=cmap(i), label=name, alpha=0.75)
    plt.legend(fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(dataset_key, encoder_name="mert_v1_330m", layer="mean_all", out_dir=None):
    cache_dir = os.path.join(CACHE_DIR, encoder_name, dataset_key)
    out_dir = out_dir or os.path.join(RESULTS_DIR, "embeddings", f"{encoder_name}_{dataset_key}")
    os.makedirs(out_dir, exist_ok=True)

    spec = DATASETS[dataset_key]
    data = load_cached(dataset_key, cache_dir, layer)
    X_tr, y_tr, _ = data["train"]
    X_val, y_val, _ = data["validation"]
    X = np.concatenate([X_tr, X_val])
    y = np.concatenate([y_tr, y_val])

    for perplexity in (15, 30, 50):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
        coords = tsne.fit_transform(X)
        plot_embedding(coords, y, spec["labels"], f"{dataset_key} t-SNE (perplexity={perplexity})",
                        os.path.join(out_dir, f"tsne_perp{perplexity}.png"))

    for n_neighbors in (10, 15, 30):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=0)
        coords = reducer.fit_transform(X)
        plot_embedding(coords, y, spec["labels"], f"{dataset_key} UMAP (n_neighbors={n_neighbors})",
                        os.path.join(out_dir, f"umap_nn{n_neighbors}.png"))

    print(f"[{dataset_key}] embedding plots saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["A", "B"], required=True)
    p.add_argument("--encoder-name", default="mert_v1_330m")
    p.add_argument("--layer", default="mean_all")
    args = p.parse_args()
    run(args.dataset, encoder_name=args.encoder_name, layer=args.layer)
