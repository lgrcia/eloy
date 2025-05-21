import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.visualization import ZScaleInterval


def z_scale(data, c=0.05):
    interval = ZScaleInterval(contrast=c)
    return interval(data.copy())


def plot_marks(
    x,
    y,
    label=None,
    position="bottom",
    offset=7,
    fontsize=12,
    color=[0.51, 0.86, 1.0],
    ms=12,
    n=None,
    inside=True,
    alpha=1,
    ax=None,
):
    y_offset = ms + offset

    if position == "top":
        y_offset *= -1

    if not isinstance(x, (list, np.ndarray, tuple)):
        x = np.array([x])
        y = np.array([y])
        if label is True:
            label = np.array([0])
        elif label is not None:
            label = np.array([label])
    else:
        if label is True:
            label = np.arange(len(x))
        elif label is not None:
            label = np.array(label)

    if ax is None:
        ax = plt.gcf().axes[0]

    if inside:
        ax = ax
        xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
        xlim.sort()
        ylim.sort()
        within = np.argwhere(
            np.logical_and.reduce([xlim[0] < x, x < xlim[1], ylim[0] < y, y < ylim[1]])
        ).flatten()
        x = x[within]
        y = y[within]
        if label is not None:
            print
            label = label[within]

    if n is not None:
        x = x[0:n]
        y = y[0:n]
        if label is not None:
            label = label[0:n]

    if label is None:
        label = [None for _ in range(len(x))]

    for _x, _y, _label in zip(x, y, label):
        circle = patches.Circle((_x, _y), ms, fill=None, ec=color, alpha=alpha)
        ax.add_artist(circle)
        f = 5
        if _label is not None:
            plt.annotate(
                _label,
                xy=[_x, _y - y_offset],
                color=color,
                ha="center",
                fontsize=fontsize,
                alpha=alpha,
                va="top" if position == "bottom" else "bottom",
            )
