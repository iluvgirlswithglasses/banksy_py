#!/usr/bin/env python
"""Compare BANKSY variants (z-scored, var-eq dense, var-eq sparse, raw)
with ground-truth annotations shown side-by-side, plus performance metrics.

Produces two figures:
  1) Spatial cluster comparison (5-panel)
  2) Performance metrics: build time, PCA+Leiden time, matrix memory

Also computes ARI (Adjusted Rand Index) for each variant against ground truth.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-banksy")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

import leidenalg

from banksy_ilgwg.matrix import (
    build_banksy_matrix,
    build_banksy_matrix_raw,
    build_banksy_matrix_vareq_dense,
    build_banksy_matrix_vareq_sparse,
)
from banksy.main import LeidenPartition
from banksy_utils.color_lists import zeileis_28

VARIANT_NAMES = [
    "z-scored\n(dense)",
    "var-eq\n(dense)",
    "var-eq\n(sparse)",
    "raw\n(sparse)",
]

VARIANT_COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]


@dataclass
class VariantResult:
    name: str = ""
    adata: ad.AnnData | None = None
    labels: np.ndarray | None = None
    n_clusters: int = 0
    modularity: float = 0.0
    ari: float | None = None
    build_sec: float = 0.0
    leiden_sec: float = 0.0
    matrix_mb: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/Tin/C04494A6.bin20_1.0.h5ad",
        help="Input AnnData .h5ad file.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/compare_variants",
        help="Directory for output plots.",
    )
    p.add_argument(
        "--ground-truth-key",
        default="leiden",
        help="Column in adata.obs containing ground-truth labels.",
    )
    p.add_argument(
        "--n-top-hvg",
        type=int,
        default=1000,
        help="Subset to this many highly-variable genes (ranked by dispersions_norm).",
    )
    p.add_argument("--lambda-param", type=float, default=0.2)
    p.add_argument("--resolution", type=float, default=0.7)
    p.add_argument("--pca-dims", type=int, default=20)
    p.add_argument("--k-geom", type=int, default=15)
    p.add_argument("--max-m", type=int, default=1)
    p.add_argument("--num-nn", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_hvg_subset(input_path: str, n_top_hvg: int) -> ad.AnnData:
    adata = ad.read_h5ad(input_path)
    adata.var_names_make_unique()

    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs.loc[:, ["x", "y"]].to_numpy()

    if "highly_variable" in adata.var:
        hvg_var = adata.var.loc[adata.var["highly_variable"].to_numpy(dtype=bool)]
        if "dispersions_norm" in hvg_var:
            hvg_var = hvg_var.sort_values(
                "dispersions_norm", ascending=False, kind="stable"
            )
        selected_genes = hvg_var.index[:n_top_hvg]
        adata = adata[:, selected_genes].copy()
    else:
        adata = adata.copy()

    adata.obsp.clear()
    adata.uns.pop("neighbors", None)
    return adata


def matrix_memory_mb(adata_obj: ad.AnnData) -> float:
    X = adata_obj.X
    if issparse(X):
        return (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    return X.nbytes / 1e6


def run_pca_leiden(
    banksy_adata: ad.AnnData,
    pca_dims: int,
    resolution: float,
    num_nn: int,
    seed: int,
) -> tuple:
    X = banksy_adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    pca = PCA(n_components=pca_dims)
    reduced = pca.fit_transform(X)

    partitioner = LeidenPartition(
        reduced,
        num_nn=num_nn,
        nns_have_weights=True,
        compute_shared_nn=True,
        filter_shared_nn=True,
        shared_nn_max_rank=3,
        shared_nn_min_shared_nbrs=5,
        verbose=False,
    )

    label, modularity = partitioner.partition(
        resolution=resolution,
        partition_metric=leidenalg.RBConfigurationVertexPartition,
        n_iterations=-1,
        seed=seed,
    )

    return label.dense.astype(int), label.num_labels, modularity


def make_colormap(num_labels: int) -> ListedColormap:
    colors = list(zeileis_28)
    if num_labels > len(colors):
        extra = plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
        colors.extend(matplotlib.colors.to_hex(c) for c in extra)
    if num_labels > len(colors):
        colors.extend(
            matplotlib.colors.to_hex(plt.get_cmap("hsv")(i / num_labels))
            for i in range(num_labels)
        )
    return ListedColormap(colors[:num_labels])


def _scatter_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    n_obs: int,
    title: str,
    n_clusters: int | None = None,
    ari: float | None = None,
) -> None:
    unique_labels = np.array(sorted(np.unique(labels)))
    remapped = (
        pd.Series(labels)
        .map({l: i for i, l in enumerate(unique_labels)})
        .to_numpy()
    )
    cmap = make_colormap(len(unique_labels))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=remapped,
        cmap=cmap,
        s=max(0.35, 40000 / n_obs),
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    pad = max(x_max - x_min, y_max - y_max) * 0.03
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    title_str = title
    if n_clusters is not None:
        title_str += f"\n{n_clusters} clusters"
    if ari is not None:
        title_str += f", ARI={ari:.4f}"
    ax.set_title(title_str, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_spatial(
    adata: ad.AnnData,
    gt_labels: np.ndarray | None,
    results: list[VariantResult],
    output_path: Path,
    lambda_param: float,
    resolution: float,
    ground_truth_key: str,
) -> None:
    coords = np.asarray(adata.obsm["spatial"])

    has_gt = gt_labels is not None
    n_panels = len(results) + (1 if has_gt else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if has_gt:
        n_gt = len(np.unique(gt_labels))
        _scatter_panel(
            axes[idx],
            coords,
            gt_labels,
            adata.n_obs,
            f"Ground Truth ({ground_truth_key})",
            n_clusters=n_gt,
        )
        idx += 1

    for r in results:
        _scatter_panel(
            axes[idx],
            coords,
            r.labels,
            adata.n_obs,
            r.name,
            n_clusters=r.n_clusters,
            ari=r.ari,
        )
        idx += 1

    fig.suptitle(
        f"BANKSY Leiden Clustering Comparison\n"
        f"lambda={lambda_param}, resolution={resolution}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spatial plot: {output_path}")


def plot_metrics(
    results: list[VariantResult],
    output_path: Path,
    lambda_param: float,
    resolution: float,
) -> None:
    names = [r.name.replace("\n", " ") for r in results]
    build_times = [r.build_sec for r in results]
    leiden_times = [r.leiden_sec for r in results]
    memories = [r.matrix_mb for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(names))
    bar_w = 0.6

    ax0 = axes[0]
    bars = ax0.bar(x, build_times, bar_w, color=VARIANT_COLORS)
    ax0.set_ylabel("Seconds")
    ax0.set_title("Build Time")
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars, build_times):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1 = axes[1]
    bars = ax1.bar(x, leiden_times, bar_w, color=VARIANT_COLORS)
    ax1.set_ylabel("Seconds")
    ax1.set_title("PCA + Leiden Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars, leiden_times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = axes[2]
    bars = ax2.bar(x, memories, bar_w, color=VARIANT_COLORS)
    ax2.set_ylabel("MB")
    ax2.set_title("Matrix Memory (adata.X)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    for bar, val in zip(bars, memories):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(memories) * 0.02,
            f"{val:.1f}MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        f"BANKSY Performance Metrics\n"
        f"lambda={lambda_param}, resolution={resolution}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot: {output_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coord_keys = ("x", "y", "spatial")
    adata = load_hvg_subset(args.input, args.n_top_hvg)
    print(f"Data: {adata.n_obs} cells x {adata.n_vars} genes")

    gt_labels = None
    if args.ground_truth_key and args.ground_truth_key in adata.obs.columns:
        gt_series = adata.obs[args.ground_truth_key]
        if hasattr(gt_series, "cat"):
            gt_labels = gt_series.cat.codes.to_numpy().astype(int)
        else:
            gt_labels = np.asarray(gt_series, dtype=int)
        n_gt = len(np.unique(gt_labels))
        print(f"Ground truth ({args.ground_truth_key}): {n_gt} classes")
    else:
        print(
            f"Warning: ground-truth key '{args.ground_truth_key}' not found in adata.obs. "
            f"Skipping ground-truth panel and ARI.",
            flush=True,
        )

    build_specs = [
        ("z-scored (dense)", build_banksy_matrix),
        ("var-eq (dense)", build_banksy_matrix_vareq_dense),
        ("var-eq (sparse)", build_banksy_matrix_vareq_sparse),
        ("raw (sparse)", build_banksy_matrix_raw),
    ]

    results: list[VariantResult] = []
    total = len(build_specs)

    for i, (name, build_fn) in enumerate(build_specs, 1):
        res = VariantResult(name=name)

        print(f"\n=== [{i}/{total*2}] Building BANKSY matrix ({name}) ===")
        t0 = time.perf_counter()
        res.adata = build_fn(
            adata,
            coord_keys,
            k_geom=args.k_geom,
            max_m=args.max_m,
            lambda_param=args.lambda_param,
        )
        res.build_sec = time.perf_counter() - t0
        res.matrix_mb = matrix_memory_mb(res.adata)
        print(
            f"  -> shape: {res.adata.shape}, "
            f"X type: {type(res.adata.X).__name__}, "
            f"memory: {res.matrix_mb:.1f} MB, "
            f"time: {res.build_sec:.1f}s"
        )

        print(f"\n=== [{i + total}/{total*2}] PCA + Leiden ({name}) ===")
        t0 = time.perf_counter()
        res.labels, res.n_clusters, res.modularity = run_pca_leiden(
            res.adata, args.pca_dims, args.resolution, args.num_nn, args.seed
        )
        res.leiden_sec = time.perf_counter() - t0
        print(
            f"  -> {res.n_clusters} clusters, "
            f"modularity={res.modularity:.4f}, "
            f"time: {res.leiden_sec:.1f}s"
        )

        results.append(res)

    if gt_labels is not None:
        for r in results:
            r.ari = adjusted_rand_score(gt_labels, r.labels)
        print("\n=== ARI ===")
        for r in results:
            print(f"  {r.name:25s}: ARI={r.ari:.4f}")

    print("\n=== Plotting spatial comparison ===")
    plot_spatial(
        adata,
        gt_labels,
        results,
        output_dir / "banksy_spatial_comparison.png",
        args.lambda_param,
        args.resolution,
        args.ground_truth_key,
    )

    print("\n=== Plotting performance metrics ===")
    plot_metrics(
        results,
        output_dir / "banksy_performance_metrics.png",
        args.lambda_param,
        args.resolution,
    )

    csv_data = {
        "cell_id": adata.obs_names,
        "x": np.asarray(adata.obsm["spatial"])[:, 0],
        "y": np.asarray(adata.obsm["spatial"])[:, 1],
    }
    for r in results:
        key = r.name.replace(" ", "_").replace("(", "").replace(")", "")
        csv_data[f"cluster_{key}"] = r.labels
    if gt_labels is not None:
        csv_data["ground_truth"] = gt_labels
    labels_csv = output_dir / "labels_comparison.csv"
    pd.DataFrame(csv_data).to_csv(labels_csv, index=False)
    print(f"Saved labels: {labels_csv}")

    metrics_rows = []
    for r in results:
        metrics_rows.append(
            {
                "variant": r.name,
                "n_clusters": r.n_clusters,
                "modularity": r.modularity,
                "ari": r.ari,
                "build_sec": round(r.build_sec, 2),
                "leiden_sec": round(r.leiden_sec, 2),
                "matrix_mb": round(r.matrix_mb, 2),
            }
        )
    metrics_csv = output_dir / "performance_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
    print(f"Saved metrics: {metrics_csv}")

    print("\n=== Summary ===")
    if gt_labels is not None:
        print(f"  Ground Truth : {len(np.unique(gt_labels))} classes")
    for r in results:
        ari_str = f", ARI={r.ari:.4f}" if r.ari is not None else ""
        print(
            f"  {r.name:25s}: {r.n_clusters} clusters, "
            f"mod={r.modularity:.4f}{ari_str}, "
            f"build={r.build_sec:.1f}s, leiden={r.leiden_sec:.1f}s, "
            f"mem={r.matrix_mb:.1f}MB"
        )


if __name__ == "__main__":
    main()
