#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:55:32 2021

@author: cdesbois
"""

import os
from datetime import datetime
from importlib import reload
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


import config


# nb description with pandas:
pd.options.display.max_columns = 30

# ===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

paths["data"] = os.path.join(paths["owncFig"], "data")


plt.rcParams.update(
    {
        "font.sans-serif": ["Arial"],
        "font.size": 14,
        "legend.fontsize": "medium",
        "figure.figsize": (11.6, 5),
        "figure.dpi": 100,
        "axes.labelsize": "medium",
        "axes.titlesize": "medium",
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "axes.xmargin": 0,
    }
)


def load_gmercier2():
    "second file containing the numerial latency values"
    file = "gmercier2.csv"
    filename = os.path.join(paths["data"], file)
    df = pd.read_csv(filename, sep="\t", decimal=",")
    df = df.set_index("cell")
    # remove empty lines
    df = df.dropna(how="all")
    return df


def check_sig(datadf):
    """
    check it the cell is in the ± sigma interval of center value
    return color series (w = non, r = left, g = right, k = both)
    """
    df = datadf.copy()
    df["sig"] = 0
    df["l-c"] = df.left - df.center
    df["r-c"] = df.right - df.center
    df["lmean"] = (df.left + df.center) / 2
    df["rmean"] = (df.right + df.center) / 2
    sigma = df.center.std()
    df.loc[df["l-c"].between(-sigma, sigma), ["sig"]] += 1
    df.loc[df["r-c"].between(-sigma, sigma), ["sig"]] += 2
    sig_colors = {0: "w", 1: "tab:red", 2: "tab:green", 3: "k"}
    df["sig_color"] = df.sig.apply(lambda x: sig_colors[x])
    ser = df.sig_color
    return df


gmdf2 = load_gmercier2()
# gmdf2['sig_color'] = check_sig(gmdf2)
gmdf2 = check_sig(gmdf2)

#%% gm latency


def hist_gm2(datadf):
    fig, axes = plt.subplots(
        ncols=1, nrows=3, figsize=(5, 14), sharex=True, sharey=True
    )
    fig.suptitle("latencies DO & D1 (gerard mercier)")
    axes = axes.flatten()
    df = datadf.copy()
    colors = ("tab:red", "tab:blue", "tab:green")
    width = (90 - 20) / (20)

    for i, col in enumerate(df.columns[:3]):
        ax = axes[i]
        height, x = np.histogram(df[col], bins=20, range=(20, 90))
        height = height / sum(height)
        x = x[:-1]
        ax.axvspan(
            df[col].mean() - df[col].std(),
            df[col].mean() + df[col].std(),
            color="tab:grey",
            alpha=0.3,
        )
        ax.axvline(df[col].mean(), color="tab:grey")
        ax.bar(
            x,
            height=height,
            width=width,
            align="edge",
            alpha=0.5,
            color=colors[i],
            edgecolor="k",
            label=col,
        )
        # ax.axvline(df[col].mean() + df[col].std(), color=colors[i], linestyle=':')
        # ax.axvline(df[col].mean() - df[col].std(), color=colors[i], linestyle=':')
        txt = "{:.1f} ± {:.1f}".format(df[col].mean(), df[col].std())
        ax.text(x=0, y=0.69, s=txt, va="top", ha="left", transform=ax.transAxes)
        txt = "{:.0f} cells".format(len(df[col].dropna()))
        ax.text(x=0, y=0.79, s=txt, va="top", ha="left", transform=ax.transAxes)
        ax.legend()
        for ax in axes:
            for spine in ["left", "top", "right"]:
                ax.spines[spine].set_visible(False)
        ax.set_yticks([])
        axes[-1].set_xlabel("time (msec)")

        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "latencies_gm.py:hist_gm2", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
fig = hist_gm2(gmdf2)
save = False
if save:
    file = "hist_gm2.pdf"
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "baudot", "latencesMercier", "fig"
    )
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%%


def scatter_gm2(datadf):
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(5, 10), sharex=True, sharey=True
    )
    fig.suptitle("latencies DO & D1 (gerard mercier)")
    axes = axes.flatten()
    df = datadf.copy()

    # left
    values = df[["center", "left", "sig_color"]].dropna().values
    x = values[:, 0]
    y = values[:, 1]
    color = values[:, 2]
    ax = axes[0]
    # ax.scatter(x, y, s=45, alpha=0.6, c='tab:red', label='left_vs_center')
    ax.scatter(
        x, y, s=45, alpha=0.6, c=color, edgecolor="tab:red", label="left_vs_center"
    )
    ax.set_ylabel("left lat. (msec)")

    # right
    values = df[["center", "right", "sig_color"]].dropna().values
    x = values[:, 0]
    y = values[:, 1]
    color = values[:, 2]
    ax = axes[1]
    # ax.scatter(x, y, s=45, alpha=0.6, c='tab:green', label='right_vs_center')
    ax.scatter(
        x, y, s=45, alpha=0.6, c=color, edgecolor="tab:green", label="right_vs_center"
    )
    ax.set_ylabel("right lat. (msec)")

    lims = (
        floor(df[df.columns[:3]].min().min() / 10.0) * 10,
        ceil(df[df.columns[:3]].max().max() / 10.0) * 10,
    )
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    for ax in axes:
        ax.plot(list(lims), list(lims), linestyle="-", color="tab:blue", alpha=0.7)
        # ax.legend()
        ax.set_xlabel("center lat. (msec)")
        # ax.set_ylabel('time (msec)')
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_gm.py:scatter_gm2",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()

    return fig


plt.close("all")
fig = scatter_gm2(gmdf2)
save = False
if save:
    file = "scatter_gm2.pdf"
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "baudot", "latencesMercier", "fig"
    )
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%% temp
df = gmdf2.copy()
sides = ["left", "right"]
i = 0

tempdf = df[sides[i]] - df.center
tempdf
tempdf["sig_colors"] = df.sig_color.values
tempdf
df.sig_color
tempdf

#%%
def diff_scatter_gm2(datadf, vsmean=True):
    """scatter plot"""
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(5, 10), sharex=True, sharey=True
    )
    fig.suptitle("latencies DO & D1 (gerard mercier)")
    axes = axes.flatten()
    df = datadf.copy()
    # left - mean
    colors = ["tab:red", "tab:green"]
    sides = ["left", "right"]
    sides = ["l-c", "r-c"]
    columns = [
        ["center", "l-c", "lmean", "sig_color"],
        ["center", "r-c", "rmean", "sig_color"],
    ]
    ylimits = []
    for i, ax in enumerate(axes):
        tempdf = df[columns[i]].dropna()
        # y = (df[sides[i]] - df.center).dropna().values
        # y = (df[[sides[i], ).dropna().values
        y = tempdf[sides[i]].values
        if vsmean:
            x = tempdf[tempdf.columns[2]].values
            xtxt = "mean value"
            xcolor = "tab:grey"
            x0 = y.mean()
        else:
            x = tempdf[tempdf.columns[0]].values
            # x = df[['center', sides[i]]].dropna().values[:,0]
            xtxt = "center"
            xcolor = "tab:blue"
            x0 = 0
        interval = (x0 - 2 * x.std(), x0 + 2 * x.std())
        ax.axhspan(*interval, alpha=0.2, color=xcolor)
        for pos in interval:
            ax.axhline(pos, color=xcolor, linewidth=2, linestyle=":", alpha=0.5)
        ax.text(
            max(x), interval[1], s="+2$\sigma$", color=xcolor, ha="left", va="bottom"
        )
        ax.text(
            max(x), interval[0], s="-2$\sigma$", color=xcolor, ha="left", va="bottom"
        )
        # ax.scatter(x, y, s=45, alpha=0.6, c=colors[i], label='l-c vs c')
        ax.scatter(
            x,
            y,
            s=55,
            alpha=0.6,
            c=tempdf.sig_color,
            edgecolor=colors[i],
            label="l-c vs c",
        )
        ax.axhline(y.mean(), color=colors[i], linewidth=3, alpha=0.5)
        txt = "{:.1f} ± {:.1f}".format(y.mean(), y.std())
        ax.text(
            x=1,
            y=0.8,
            s=txt,
            va="top",
            ha="right",
            color=colors[i],
            transform=ax.transAxes,
        )
        ax.axvline(x.mean(), color=xcolor, linewidth=3, alpha=0.5)
        txt = "{:.1f} ± {:.1f} msec".format(x.mean(), x.std())
        ax.text(
            x=1,
            y=0.7,
            s=txt,
            va="top",
            ha="right",
            color=xcolor,
            transform=ax.transAxes,
        )
        ylimits.append((y.min(), y.max()))
        ax.set_xlabel(xtxt, color=xcolor)
        ax.set_ylabel("{} - center (msec)".format(sides[i]), color=colors[i])
        # regression
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        ax.plot(
            x, regr.predict(x), color=colors[i], linestyle=":", linewidth=3, alpha=0.5
        )

    # adjust limits
    lims = (floor(x.min() / 10.0) * 10, ceil(x.max() / 10.0) * 10)
    ax.set_xlim(lims)
    lims = (floor(min(min(ylimits)) / 10.0) * 10, ceil(max(max(ylimits)) / 10.0) * 10)
    ax.set_ylim(lims)

    for ax in axes:
        ax.axhline(0, color="tab:grey")
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_gm.py:diff_scatter_gm2",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
fig1 = diff_scatter_gm2(gmdf2)
fig2 = diff_scatter_gm2(gmdf2, vsmean=False)
save = False
if save:
    file = "diffmean_scatter_gm2.pdf"
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "baudot", "latencesMercier", "fig"
    )
    filename = os.path.join(dirname, file)
    fig1.savefig(filename)

    file = "diff_scatter_gm2.pdf"
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "baudot", "latencesMercier", "fig"
    )
    filename = os.path.join(dirname, file)
    fig2.savefig(filename)

#%%


def hist_diff_gm(datadf):
    """hitogramm of latency distribution"""
    fig, axes = plt.subplots(
        ncols=1, nrows=2, figsize=(5, 10), sharex=True, sharey=True
    )
    fig.suptitle("latencies DO & D1 (gerard mercier)")
    axes = axes.flatten()

    df = pd.DataFrame()
    df["l-c"] = datadf.left - datadf.center
    df["r-c"] = datadf.right - datadf.center
    colors = ("tab:red", "tab:green")
    width = (90 - 20) / (20)

    for i, col in enumerate(df.columns):
        width = (df[col].max() - df[col].min()) / len(df[col].dropna())
        width = 2
        ax = axes[i]
        height, x = np.histogram(df[col].dropna(), bins=20)
        height = height / sum(height)
        x = x[:-1]
        ax.bar(
            x,
            height=height,
            width=width,
            align="edge",
            alpha=0.5,
            color=colors[i],
            edgecolor="k",
            label=col.replace("-", " - "),
        )
        ax.axvline(df[col].mean(), color=colors[i], linewidth=3, alpha=0.5)
        txt = "diff: {:.1f} ± {:.1f}".format(df[col].mean(), df[col].std())
        ax.text(x=0, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
        txt = "{:.0f} cells".format(len(df[col].dropna()))
        ax.text(x=0, y=0.9, s=txt, va="top", ha="left", transform=ax.transAxes)
        ax.legend()
        # interval
        positions = [-2 * datadf.center.std(), 2 * datadf.center.std()]
        for ax in axes:
            ax.axvline(0, color="tab:blue")
            for spine in ["left", "top", "right"]:
                ax.spines[spine].set_visible(False)
            for pos in positions:
                ax.axvline(pos, color="tab:blue", linewidth=2, linestyle=":", alpha=0.5)
                ax.text(
                    pos,
                    ax.get_ylim()[1],
                    s="2$\sigma$",
                    color="tab:blue",
                    ha="left",
                    va="top",
                )
        ax.set_yticks([])
        axes[-1].set_xlabel("time (msec)")
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_gm.py:histdiff_gm2",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
fig = hist_diff_gm(gmdf2)
save = False
if save:
    file = "hist_diff_gm.pdf"
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "baudot", "latencesMercier", "fig"
    )
    filename = os.path.join(dirname, file)
    fig.savefig(filename)
