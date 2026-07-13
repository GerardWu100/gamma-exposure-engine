"""Generate the technical charts used by the gamma-exposure blog post.

The script reads only frozen pipeline artifacts stored beside this file. It
writes deterministic PNG figures into ``blog/images`` and does not access the
network or the project's raw inputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BLOG_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BLOG_DIR / "data"
IMAGE_DIR: Path = BLOG_DIR / "images"
FIGURE_DPI: int = 180
FIGURE_SIZE: tuple[float, float] = (11.0, 6.5)
GAMMA_SCALE: float = 1.0e12
VARIANCE_SCALE: float = 1.0e4
POINT_SIZE: int = 72
BOOTSTRAP_ALPHA: float = 0.22

NAVY: str = "#0b1f33"
CYAN: str = "#1f9eaa"
AMBER: str = "#d9902f"
PALE_BLUE: str = "#dceef2"
GRID: str = "#c9d2d8"


def configure_plot_style() -> None:
    """Apply a readable, restrained style shared by all blog figures."""

    plt.rcParams.update(
        {
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "font.size": 11,
            "figure.facecolor": "white",
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "savefig.facecolor": "white",
            "xtick.color": NAVY,
            "ytick.color": NAVY,
        }
    )


def plot_daily_alignment() -> None:
    """Plot day-t gamma beside the variance observed on the next trade day."""

    frame = pl.read_parquet(DATA_DIR / "research_dataset.parquet").sort("trade_date")
    dates = frame.get_column("trade_date").to_list()
    gamma = (
        frame.get_column("total_open_interest_weighted_gamma").to_numpy() / GAMMA_SCALE
    )
    variance = (
        frame.get_column("next_day_realized_variance").to_numpy() * VARIANCE_SCALE
    )

    figure, gamma_axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    variance_axis = gamma_axis.twinx()
    gamma_line = gamma_axis.plot(
        dates,
        gamma,
        color=CYAN,
        marker="o",
        linewidth=2.2,
        label="Day-t gamma mass",
    )[0]
    variance_line = variance_axis.plot(
        dates,
        variance,
        color=AMBER,
        marker="s",
        linewidth=2.0,
        label="Next-day realized variance",
    )[0]

    gamma_axis.set_title("SPY gamma mass and the following day's realized variance")
    gamma_axis.set_xlabel("Exposure date (2024)")
    gamma_axis.set_ylabel("Open-interest-weighted gamma (trillions)", color=CYAN)
    variance_axis.set_ylabel("Realized variance (×10⁻⁴)", color=AMBER)
    gamma_axis.grid(axis="y")
    gamma_axis.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    gamma_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    gamma_axis.tick_params(axis="x", rotation=35)
    gamma_axis.legend(
        [gamma_line, variance_line],
        [gamma_line.get_label(), variance_line.get_label()],
        loc="upper left",
        frameon=False,
    )
    figure.savefig(IMAGE_DIR / "01-daily-alignment.png", dpi=FIGURE_DPI)
    plt.close(figure)


def plot_quantile_summary() -> None:
    """Plot quintile mean variance and its frozen bootstrap interval."""

    frame = pl.read_csv(DATA_DIR / "quantile_summary.csv").sort("quantile_bucket")
    buckets = frame.get_column("quantile_bucket").to_numpy() + 1
    means = frame.get_column("target_mean").to_numpy() * VARIANCE_SCALE
    lower = frame.get_column("ci_lower").to_numpy() * VARIANCE_SCALE
    upper = frame.get_column("ci_upper").to_numpy() * VARIANCE_SCALE
    errors = np.vstack((means - lower, upper - means))

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.bar(buckets, means, color=CYAN, width=0.68, alpha=0.88)
    axis.errorbar(
        buckets,
        means,
        yerr=errors,
        fmt="none",
        ecolor=NAVY,
        elinewidth=1.7,
        capsize=5,
    )
    axis.set_title(
        "Next-day variance does not increase monotonically across gamma quintiles"
    )
    axis.set_xlabel("Open-interest-weighted gamma quintile (low to high)")
    axis.set_ylabel("Mean next-day realized variance (×10⁻⁴)")
    axis.set_xticks(buckets)
    axis.grid(axis="y")
    axis.text(
        0.99,
        0.97,
        "95% percentile bootstrap intervals; 4 observations per quintile",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color=NAVY,
    )
    figure.savefig(IMAGE_DIR / "02-quantile-variance.png", dpi=FIGURE_DPI)
    plt.close(figure)


def plot_factor_scatter() -> None:
    """Plot the aligned factor-target observations with a descriptive fit line."""

    frame = pl.read_parquet(DATA_DIR / "research_dataset.parquet")
    gamma = (
        frame.get_column("total_open_interest_weighted_gamma").to_numpy() / GAMMA_SCALE
    )
    variance = (
        frame.get_column("next_day_realized_variance").to_numpy() * VARIANCE_SCALE
    )
    fit_slope, fit_intercept = np.polyfit(gamma, variance, deg=1)
    fit_x = np.linspace(float(gamma.min()), float(gamma.max()), 100)
    fit_y = fit_intercept + fit_slope * fit_x

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.scatter(
        gamma,
        variance,
        s=POINT_SIZE,
        color=CYAN,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    axis.plot(fit_x, fit_y, color=AMBER, linewidth=2.0, label="Descriptive linear fit")
    axis.set_title("Twenty observations show little gamma-mass/variance association")
    axis.set_xlabel("Day-t open-interest-weighted gamma (trillions)")
    axis.set_ylabel("Next-day realized variance (×10⁻⁴)")
    axis.grid()
    axis.legend(frameon=False)
    axis.text(
        0.03,
        0.95,
        "Spearman ρ = 0.030\np = 0.900\nn = 20",
        transform=axis.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": PALE_BLUE,
            "edgecolor": "none",
        },
    )
    figure.savefig(IMAGE_DIR / "03-factor-scatter.png", dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """Generate all post figures from the frozen analysis artifacts."""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_daily_alignment()
    plot_quantile_summary()
    plot_factor_scatter()


if __name__ == "__main__":
    main()
