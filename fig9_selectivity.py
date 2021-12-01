#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:41:03 2021

@author: cdesbois
"""

import os
from datetime import datetime

from importlib import reload
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import fig_proposal as figp
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

paths = config.build_paths()
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "fillinIn", "indFill_popFill"
)


#%%
plt.close("all")


def plot_fill_combi(data_fill, data_pop, stdcolors=std_colors, anot=anot):

    # to build
    pop2sigdf = pop2sig_df
    popfilldf = popfill_df

    # data_fill = pop2sigdf

    add_std = True
    stdcolors = config.std_colors()
    colors = [stdcolors[st] for st in ["k", "red", "green", "yellow", "blue", "blue"]]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    # fill pop
    filldf = popfilldf.copy()

    # general  pop
    gen_df = pop2sigdf.copy()
    # defined in dataframe columns (first column = ctr))
    kind, rec, spread, *_ = gen_df.columns.to_list()[1].split("_")
    # centering
 #   middle = (gen_df.index.max() - gen_df.index.min()) / 2
  #  gen_df.index = (gen_df.index - middle) / 10
    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    # subtract the centerOnly response (ref = df['CENTER-ONLY'])
    ref = gen_df[gen_df.columns[0]]
    gen_df = gen_df.subtract(ref, axis=0)
    # remove rdsect
    cols = gen_df.columns.to_list()
    while any(_ for _ in cols if "sect_rd" in _):
        cols.remove(next(_ for _ in cols if "sect_rd" in _)
    # buils labels
    labels = cols[:]
    labels = [n.replace("full_rd_", "full_rdf_") for n in labels]
    for i in range(3):
        for item in labels:
            if len(item.split("_")) < 6:
                j = labels.index(item)
                labels[j] = item + "_ctr"
    labels = [st.split("_")[-3] for st in labels]

    fig = plt.figure(figsize=(11.6, 8))
    axes = []
    ax = fig.add_subplot(221)
    axes.append(ax)
    ax1 = fig.add_subplot(223, sharex=ax)
    axes.append(ax1)
    ax = fig.add_subplot(222)
    axes.append(ax)
    ax1 = fig.add_subplot(224, sharex=ax, sharey=ax)
    axes.append(ax1)

    # fill pop
    spks = df.columns[13:18]
    vms = [df.columns[i] for i in [0, 1, 9, 10, 11]]

    # vm pop
    ax = axes[0]
    for i, col in enumerate(vms):
        ax.plot(
            df[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=1.5,
            label=df.columns[i],
        )
        if add_std:
            if col == "popfillVmscpIsoStc":
                ax.fill_between(
                    df.index,
                    df.popfillVmscpIsoStcSeup,
                    df.popfillVmscpIsoStcSedw,
                    color=colors[i],
                    alpha=0.3,
                )

    ax.set_xlim(-20, 50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.set_ylabel("Normalized Vm")
    ax.annotate(
        "n=12", xy=(0.1, 0.8), size="large", xycoords="axes fraction", ha="center"
    )
    ax.annotate(
        "Surround-then-Center",
        xy=(1, 1),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    # spk pop
    ax = axes[1]
    for i, col in enumerate(spks):
        ax.plot(
            df[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=1.5,
            label=df.columns[i],
        )
        if add_std:
            pass
    x = 0
    y = df[spks[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.set_xlim(-20, 50)
    ax.set_ylabel("Normalized Firing Rate")
    ax.annotate(
        "n=7", xy=(0.1, 0.8), size="large", xycoords="axes fraction", ha="center"
    )
    ax.set_xlabel("Relative Time (ms)")
    ax.annotate(
        "Surround-then-Center",
        xy=(1, 1),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    # surround only
    ax = axes[2]
    surround_cols = [df.columns[st] for st in (2, 19, 20, 21)]
    # +1 because no black curve
    for i, col in enumerate(surround_cols):
        # for i in (2,19,20,21):
        ax.plot(
            df[col], color=colors[i + 1], alpha=alphas[i + 1], linewidth=1.5, label=col
        )
        if add_std:
            if col == "popfillVmscpIsoSo":
                ax.fill_between(
                    df.index,
                    df.popfillVmscpIsoSoSeup,
                    df.popfillVmscpIsoSoSedw,
                    color=colors[i + 1],
                    alpha=0.3,
                )
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    # vspread = .06  # vertical spread for realign location
    # ax1.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-150, 150)

    ax.set_ylabel("Normalized Vm")
    ax.annotate(
        "n=12", xy=(0.1, 0.8), size="large", xycoords="axes fraction", ha="center"
    )
    ax.annotate(
        "Surround",
        xy=(1, 1),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    # gen population
    ax = axes[3]
    cols = gen_df.columns
    for i, col in enumerate(cols[:-1]):
        ax.plot(
            gen_df[col],
            color=colors[i],
            alpha=alphas[i],
            label=labels[i],
            linewidth=1.5,
        )
    # max_x center only
    ax.axvline(21.4, alpha=0.4, color="k")
    # end_x of center only
    # (df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
    ax.axvline(88, alpha=0.3)
    ax.axvspan(0, 88, facecolor="k", alpha=0.1)

    # ax.text(0.50, 0.88, 'center only response \n start | peak | end',
    #         transform=ax.transAxes, alpha=0.5)
    ax.set_ylabel(r"$\Delta$ Normalized Vm")
    ax.annotate(
        "n=15", xy=(0.1, 0.8), size="large", xycoords="axes fraction", ha="center"
    )
    ax.set_xlabel("Relative Time (ms)")
    # ax.annotate("SurroundThenCenter minus Center", xy=(1, 1), size='large',
    #              xycoords="axes fraction", ha='right', va='top')
    # bbox=dict(fc=(1, 1, 1), ec=(1, 1, 1)))
    ax.annotate(
        "Surround-then-Center $\it{minus}$ Center",
        xy=(1, 1),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    for ax in axes:
        ax.axhline(0, alpha=0.3, color="k")
        ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    # align zero between subplots
    # gfunc.align_yaxis(ax1, 0, ax2, 0)
    for ax in axes[:2]:
        ax.set_xlim(-20, 60)
        custom_ticks = np.arange(-20, 60, 10)[1:]
        ax.set_xticks(custom_ticks)
        if ax != axes[1]:
            ax.set_ylim(-0.1, 1.4)
            custom_ticks = np.arange(0, 1.1, 0.2)
            ax.set_yticks(custom_ticks)
        elif ax == axes[1]:
            # ax.set_ylim(-.1, 1.4)
            ax.set_ylim(-0.1, 1.1)
            custom_ticks = np.arange(0, 1.1, 0.2)
            ax.set_yticks(custom_ticks)
    for ax in axes[2:]:
        ax.set_xlim(-150, 150)
        ax.set_ylim(-0.15, 0.4)
        ax.set_xticks(np.linspace(-150, 150, 7)[1:-1])
        ax.set_yticks(np.linspace(-0.1, 0.3, 5)[1:])

    gfunc.align_yaxis(axes[2], 0, axes[0], 0)
    gfunc.align_yaxis(axes[3], 0, axes[1], 0)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fill.py:plot_fill_combi", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        fig.text(0.5, 0.01, "summary", ha="center", va="bottom", alpha=0.4)

    return fig


plt.close("all")
# general pop
select = dict(age="new", rec="vm", kind="sig")
# select['align'] = 'p2p'

data_df, file = ltra.load_intra_mean_traces(paths, **select)

fig = plot_fill_combi(data_fill=pop_df, data_pop=data_df)

save = False
if save:
    folder = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    file = "fill_combi"
    filename = os.path.join(folder, (file + "png"))
    fig.savefig(filename)
    # update current
    file = "f9_" + file
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)