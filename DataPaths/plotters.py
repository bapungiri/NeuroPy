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

colors = ["#999897", "#f07067"] + ["#424242", "#eb4034"] * 2 + ["#424242", "#48bdf7"]


def violinplot(data, x, y, hue, hue_order, ax=None, **kwargs):
    plot_kw = dict(data=data, x=x, y=y, hue=hue, hue_order=hue_order, ax=ax)
    sns.violinplot(
        **plot_kw,
        split=True,
        inner="quartile",
        linewidth=0,
        palette=colors_sd(1),
        # colors=colors,
        # scale="width",
        saturation=1,
        cut=True,
        **kwargs
    )
    # sns.pointplot(
    #     **plot_kw,
    #     palette=["w", "w"],
    #     # palette=["#fafa23", "#fafa23"],
    #     dodge=0.3,
    #     ci=None,
    #     join=False,
    #     markers="o",
    #     scale=0.3
    # )
    for p in ax.lines:
        p.set_linestyle("-")
        p.set_linewidth(0.5)  # Sets the thickness of the quartile lines
        p.set_color("white")  # Sets the color of the quartile lines
        p.set_alpha(1)

    return ax
