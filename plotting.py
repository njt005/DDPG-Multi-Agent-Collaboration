import typing as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """moving_average
    Calculates the moving average of the input array

    Parameters
    ----------
    a : np.ndarray
    n : int
        moving average window length

    Returns
    -------
    np.ndarray
        Moving average of input array
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def create_reward_plot(
    scores: np.ndarray,
    save_path: t.Optional[str] = None,
) -> None:
    """create_reward_plot
    Create plot of rewards during training

    Parameters
    ----------
    scores : np.ndarray
        rewards that agent received
    smooth_n : t.Optional[int], optional
        moving average bin length to smooth curve
    save_path : t.Optional[str], optional
        path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    scores_m = np.mean(scores, axis=0)
    episodes = np.arange(len(scores_m))
    for score in scores:
        ax.plot(episodes, score, c="k", lw=0.5)
    ax.plot(episodes, scores_m, c="k", lw=2)

    ax.set_ylabel("Score")
    ax.set_xlabel("Episode")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")


def create_seaborn_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: t.Optional[str] = None,
    colors=None,
    title: str = None,
    figsize=(6, 4),
    ascending: bool = True,
    plot_mean: bool = True,
    save_path: t.Optional[str] = None,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    if ascending:
        group_order = df.groupby(x)[y].agg("mean").sort_values().index
    else:
        group_order = None

    if hue is not None:
        palette = None
    else:
        if colors is not None:
            palette = colors
        else:
            palette = ["#6699CC"]

    sns.barplot(
        data=df,
        x=x,
        y=y,
        estimator=np.mean,
        order=group_order,
        palette=palette,
        hue=hue,
        zorder=1,
        ax=ax,
    )

    if plot_mean:
        y_m = df[y].mean()
        y_s = df[y].std()

        ax.hlines(
            y=[y_m],
            xmin=[-0.75],
            xmax=[len(df) - 0.25],
            linestyles=["dashed"],
            colors=["k"],
        )
        ax.text(-0.5, y_m, rf"{y_m:.0f}$\pm${y_s:.0f} episodes", ha="left", va="bottom")
        ax.set_xlim([-0.75, len(df) - 0.25])

    if title is not None:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
