#!/usr/bin/env python3
"""Create a Figure-2-style QSpec probability plot from a token trace CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


W4A16 = "w4a16_top1_probability"
W4A4 = "w4a4_top1_probability"
MATCH = "top1_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot W4A4/W4A16 token probabilities like QSpec Figure 2."
    )
    parser.add_argument("csv", type=Path, help="qspec_token_trace.csv")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("qspec_figure2_reproduction")
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def marginal_hist(
    axis: plt.Axes,
    accepted: np.ndarray,
    rejected: np.ndarray,
    bins: np.ndarray,
    orientation: str,
) -> None:
    accepted_hist, edges = np.histogram(accepted, bins=bins, density=True)
    rejected_hist, _ = np.histogram(rejected, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    if orientation == "vertical":
        axis.fill_between(
            centers, accepted_hist, color="#238b45", alpha=0.30, linewidth=0
        )
        axis.plot(centers, accepted_hist, color="#238b45", linewidth=1.4)
        axis.plot(centers, rejected_hist, color="#cb181d", linewidth=1.4)
    else:
        axis.fill_betweenx(
            centers, 0, accepted_hist, color="#238b45", alpha=0.30, linewidth=0
        )
        axis.plot(accepted_hist, centers, color="#238b45", linewidth=1.4)
        axis.plot(rejected_hist, centers, color="#cb181d", linewidth=1.4)


def make_plot(df: pd.DataFrame, threshold: float) -> plt.Figure:
    df = df[[W4A16, W4A4, MATCH]].dropna().copy()
    df[MATCH] = bool_series(df[MATCH])

    accepted = df[df[MATCH]]
    rejected = df[~df[MATCH]]
    total = len(df)
    agreement = len(accepted) / total
    both_high = (df[W4A16] > threshold) & (df[W4A4] > threshold)
    high_agreement = df.loc[both_high, MATCH].mean()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )
    figure = plt.figure(figsize=(8.2, 7.6), constrained_layout=False)
    grid = GridSpec(
        4,
        4,
        figure=figure,
        left=0.11,
        right=0.95,
        bottom=0.10,
        top=0.91,
        hspace=0.05,
        wspace=0.05,
    )
    top = figure.add_subplot(grid[0, :3])
    main = figure.add_subplot(grid[1:, :3], sharex=top)
    right = figure.add_subplot(grid[1:, 3], sharey=main)

    # Draw accepted tokens first and rejected tokens last so rare mismatches
    # remain visible in the dense upper-right region.
    main.scatter(
        accepted[W4A16],
        accepted[W4A4],
        s=3,
        alpha=0.035,
        c="#238b45",
        edgecolors="none",
        rasterized=True,
    )
    main.scatter(
        rejected[W4A16],
        rejected[W4A4],
        s=5,
        alpha=0.38,
        c="#cb181d",
        edgecolors="none",
        rasterized=True,
    )

    main.axvline(threshold, color="#555555", linestyle="--", linewidth=1)
    main.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
    main.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=0.8)
    main.set_xlim(0, 1.01)
    main.set_ylim(0, 1.01)
    main.set_xlabel("W4A16 top-1 token probability")
    main.set_ylabel("W4A4 top-1 token probability")
    main.grid(color="#dddddd", linewidth=0.5, alpha=0.65)

    bins = np.linspace(0, 1, 101)
    marginal_hist(
        top,
        accepted[W4A16].to_numpy(),
        rejected[W4A16].to_numpy(),
        bins,
        "vertical",
    )
    marginal_hist(
        right,
        accepted[W4A4].to_numpy(),
        rejected[W4A4].to_numpy(),
        bins,
        "horizontal",
    )
    top.axvline(threshold, color="#555555", linestyle="--", linewidth=1)
    right.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
    top.set_ylabel("Density")
    right.set_xlabel("Density")
    top.tick_params(axis="x", labelbottom=False)
    right.tick_params(axis="y", labelleft=False)
    top.spines[["top", "right"]].set_visible(False)
    right.spines[["top", "right"]].set_visible(False)

    legend = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color="#238b45",
            label=f"Accepted ({len(accepted):,})", markersize=6,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", color="#cb181d",
            label=f"Rejected ({len(rejected):,})", markersize=6,
        ),
        Line2D(
            [0], [0], color="#555555", linestyle="--",
            label=f"Probability = {threshold:.1f}",
        ),
    ]
    main.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.92)

    figure.suptitle(
        "Token prediction probabilities of W4A4 and W4A16",
        y=0.975,
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.11,
        0.935,
        (
            f"GSM8K · {total:,} draft tokens · Top-1 agreement "
            f"{agreement:.2%} · Agreement when both > {threshold:.0%}: "
            f"{high_agreement:.3%}"
        ),
        fontsize=9.5,
        color="#333333",
    )
    return figure


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.csv)
    figure = make_plot(data, args.threshold)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
