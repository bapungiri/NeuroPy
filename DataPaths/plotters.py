import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from subjects import colors_sd

sns_violin_kw = dict(
    palette=colors_sd(1),
    saturation=1,
    linewidth=0.4,
    cut=True,
    split=False,
    inner="box",
    showextrema=False,
    # showmeans=True,
)


def violinplot(data, x, y, hue, hue_order, ax=None, **kwargs):
    plot_kw = dict(data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=ax)
    sns.violinplot(
        **plot_kw,
        split=True,
        inner=None,
        linewidth=0.4,
        palette=colors_sd(1),
        saturation=1,
        cut=True,
        **kwargs
    )
    sns.pointplot(
        **plot_kw,
        palette=["w", "w"],
        dodge=0.3,
        ci=None,
        join=False,
        markers=".",
        scale=0.3
    )
    return ax
