"""Regenerate the paper's figures.

Run `python src/figures.py` after `pipeline.py`. Writes to results/figures/.

The projection is recomputed here with a fixed seed rather than stored, so both
figures are drawn on the same coordinates and either can be reproduced from the
raw captures alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
import pipeline as P  # noqa: E402

# Palette. Figure 1 uses a diverging pair (more/less than control); Figure 2
# uses a single sequential hue so the two do not collide in meaning.
DIVERGING = ["#eb6834", "#f7f7f5", "#2a78d6"]   # less | same | more
SEQUENTIAL = ["#ffffff", "#c9c3e6", "#4a3aa7"]
CONTEXT = ["#ffffff", "#b9b9b4"]
POINT = "#2b2b2b"
RULE = "#c9c9c5"
INK, MUTED = "#0b0b0b", "#52514e"

LABEL = {
    "watch": "watch (control)",
    "like": "like",
    "notInt": "not-interested",
    "combined": "combined",
}
GRID = 200
BANDWIDTH = 0.28


def project(corpus: pd.DataFrame) -> pd.DataFrame:
    """PCA then t-SNE, seeded, matching the parameters in config."""
    x = np.stack(corpus["embedding"].values).astype(np.float64)
    reduced = PCA(n_components=C.PCA_COMPONENTS, random_state=42).fit_transform(x)
    coords = TSNE(
        n_components=2,
        perplexity=C.TSNE_PERPLEXITY,
        learning_rate=C.TSNE_LEARNING_RATE,
        random_state=42,
        init="pca",
    ).fit_transform(reduced)
    out = corpus.copy()
    out["z1"], out["z2"] = coords[:, 0], coords[:, 1]
    return out


def _grid(frame):
    x, y = frame["z1"].values, frame["z2"].values
    gx, gy = np.mgrid[x.min():x.max():complex(GRID), y.min():y.max():complex(GRID)]
    return gx, np.vstack([gx.ravel(), gy.ravel()]), [x.min(), x.max(), y.min(), y.max()]


def _density(subset, points, shape):
    kde = gaussian_kde(np.vstack([subset["z1"], subset["z2"]]), bw_method=BANDWIDTH)
    return kde(points).reshape(shape)


def _bare(ax, extent):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.6)


def difference_figure(frame: pd.DataFrame, path):
    """Where each condition is denser or sparser than the control."""
    gx, points, extent = _grid(frame)
    control = frame[frame["condition"] == "watch"]
    base = _density(control, points, gx.shape)
    others = ["like", "notInt", "combined"]
    diffs = {c: _density(frame[frame["condition"] == c], points, gx.shape) - base
             for c in others}
    limit = max(np.abs(v).max() for v in diffs.values())

    div = LinearSegmentedColormap.from_list("div", DIVERGING)
    grey = LinearSegmentedColormap.from_list("grey", CONTEXT)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.12), sharex=True, sharey=True)

    def scatter(ax, subset):
        ax.scatter(subset["z1"], subset["z2"], s=0.7, c=POINT,
                   alpha=0.38, linewidths=0, rasterized=True)

    axes[0].imshow(base.T, origin="lower", extent=extent, cmap=grey, aspect="auto")
    scatter(axes[0], control)
    axes[0].set_title(LABEL["watch"], fontsize=8, color=INK, pad=4)
    for ax, cond in zip(axes[1:], others):
        ax.imshow(diffs[cond].T, origin="lower", extent=extent, cmap=div,
                  vmin=-limit, vmax=limit, aspect="auto")
        scatter(ax, frame[frame["condition"] == cond])
        ax.set_title(LABEL[cond], fontsize=8, color=INK, pad=4)
    for ax in axes:
        _bare(ax, extent)

    fig.subplots_adjust(wspace=0.07)
    mappable = plt.cm.ScalarMappable(cmap=div)
    mappable.set_array([])
    bar = fig.colorbar(mappable, ax=axes, fraction=0.017, pad=0.010, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["less than\ncontrol", "same", "more than\ncontrol"],
                           fontsize=6, color=MUTED)
    bar.outline.set_visible(False)
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def category_figure(frame: pd.DataFrame, path):
    """Where each content category sits, on the same projection."""
    gx, points, extent = _grid(frame)
    cats = [("mainstream", "mainstream"),
            ("generic", "generic gaming"),
            ("puzzle", "puzzle gaming")]
    dens = {k: _density(frame[frame["category"] == k], points, gx.shape) for k, _ in cats}
    vmax = max(v.max() for v in dens.values())

    seq = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), sharex=True, sharey=True)
    for ax, (key, label) in zip(axes, cats):
        ax.imshow(dens[key].T, origin="lower", extent=extent, cmap=seq,
                  vmin=0, vmax=vmax, aspect="auto")
        subset = frame[frame["category"] == key]
        ax.scatter(subset["z1"], subset["z2"], s=0.9, c=POINT,
                   alpha=0.34, linewidths=0, rasterized=True)
        ax.set_title(label, fontsize=8.2, color=INK, pad=12)
        ax.text(0.5, 1.015, f"{100 * len(subset) / len(frame):.0f}% of corpus",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.2, color=MUTED)
        _bare(ax, extent)

    fig.subplots_adjust(wspace=0.07)
    mappable = plt.cm.ScalarMappable(cmap=seq)
    mappable.set_array([])
    bar = fig.colorbar(mappable, ax=axes, fraction=0.017, pad=0.010, ticks=[0, 1])
    bar.ax.set_yticklabels(["sparse", "dense"], fontsize=6.5, color=MUTED)
    bar.outline.set_visible(False)
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    corpus = P.attach_categories(
        P.align_timestamps(P.load_embeddings(), P.wrangle(P.load_raw(), "published"))
    )
    print(f"projecting {len(corpus)} rows (PCA {C.PCA_COMPONENTS} -> t-SNE, seed 42)")
    frame = project(corpus)

    difference_figure(frame, C.FIGURES / "tsne_difference.png")
    print(f"wrote {C.FIGURES / 'tsne_difference.png'}")
    category_figure(frame, C.FIGURES / "tsne_categories.png")
    print(f"wrote {C.FIGURES / 'tsne_categories.png'}")


if __name__ == "__main__":
    main()
