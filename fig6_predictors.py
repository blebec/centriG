#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:29:03 2021

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

paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)

#%% figure 6


def build_fig6_predictors_datafile(write=False):
    """ combine the xcel files to build fig6 dataframes
        input:
            write : boolean to save the data
        output:
            indidf = pd.dataframe for individual example
            popdf = pd.dataframe for population
            """
    indidf = ldat.load_filldata("indi")
    popdf = ldat.load_filldata("pop")

    # manage supplementary data (variability)
    supfile = "fig6_supData.xlsx"
    supfilename = os.path.join(paths["sup"], supfile)
    supdf = pd.read_excel(supfilename)
    # df = sup_df.copy()
    # centering
    middle = (supdf.index.max() - supdf.index.min()) / 2
    supdf.index = (supdf.index - middle) / 10
    # limit the date time range
    supdf = supdf.loc[-200:200]

    # individual
    icols = [_ for _ in supdf.columns if _[:4].isdigit()]
    pcols = [_ for _ in supdf.columns if not _[:4].isdigit()]
    # get cell name
    key = {"_".join(_.split("_")[:2]) for _ in icols}.pop()

    # format indi_df
    # indidf = indi_df.copy()
    indicols = indidf.columns
    indicols = [_.lower() for _ in indicols]
    indicols = [_.strip("s") for _ in indicols]
    indicols = [_.replace("cpiso", "cpiso_") for _ in indicols]
    indicols = [key + "_" + _ for _ in indicols]
    indidf.columns = indicols

    # join
    vardf = supdf[icols]
    vardf.columns = [_.replace("_ctr_stc_", "_ctr_") for _ in vardf.columns]
    indidf = indidf.join(vardf)

    # pop_df
    # popdf = pop_df.copy()
    popcols = popdf.columns
    popcols = [_.replace("popfill", "popfill_") for _ in popcols]
    popcols = [_.replace("_Vm", "_Vm_") for _ in popcols]
    popcols = [_.replace("_Spk", "_Spk_") for _ in popcols]
    popcols = [_.replace("_scpIso", "_s_cpiso_") for _ in popcols]
    popcols = [_.replace("_scfIso", "_s_cfiso_") for _ in popcols]
    popcols = [_.replace("_frnd", "_f_rnd_") for _ in popcols]
    popcols = [_.replace("_srnd", "_s_rnd_") for _ in popcols]
    popcols = [_.replace("_scpCross", "s_cpcx_") for _ in popcols]
    popcols = [_.replace("_Vms", "_Vm_s") for _ in popcols]
    popcols = [_.replace("_Spks", "_Spk_S") for _ in popcols]
    popcols = [_.replace("_S_", "_s_") for _ in popcols]
    popcols = [_.replace("_Ctr", "_ctr") for _ in popcols]
    popcols = [_.replace("rnd_Iso", "rnd_") for _ in popcols]
    popcols = [_.replace("_Stc", "_stc_") for _ in popcols]
    popcols = [_.replace("_So", "_so_") for _ in popcols]
    popcols = [_.replace("__", "_") for _ in popcols]
    popcols = [_.strip("_") for _ in popcols]
    popdf.columns = popcols

    # join
    vardf = supdf[pcols]
    varcols = vardf.columns
    varcols = [_.replace("pop_fillsig_", "popfill_Vm_s_") for _ in varcols]
    varcols = [_.replace("_s_ctr_stc_", "_ctr_") for _ in varcols]
    vardf.columns = varcols
    popdf = popdf.join(vardf)

    # datasavename = os.path.join(paths["sup"], "fig6s.hdf")
    savefile = "fig6s.hdf"
    keys = ["indi", "pop"]
    dfs = [indidf, popdf]
    savedirname = paths["figdata"]
    savefilename = os.path.join(savedirname, savefile)
    for key, df in zip(keys, dfs):
        print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
        for item in df.columns:
            print(item)
        print()
        if write:
            df.to_hdf(savefilename, key)

    return indidf, popdf


def load_fig6_predictors_datafile():
    """ load the indidf and popdf dataframe for fig 6 """
    loadfile = "fig6s.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, loadfile)
    indidf = pd.read_hdf(loadfilename, "indi")
    popdf = pd.read_hdf(loadfilename, "pop")

    return indidf, popdf


def plot_fig6_predictors(inddata, popdata, stdcolors=std_colors, anot=True):
    """
    filling in example + pop
    """
    # boolean to add std
    indi_std = True  # not used at the moment (no data available)
    pop_std = True
    # ## to build
    # inddata = indi_df
    # popdata = pop_df
    # stdcolors=std_colors
    # anot=True
    # ##
    legend = False
    # idf = inddata.copy()
    # cols = [
    #     "Center-Only",
    #     "Surround-then-Center",
    #     "Surround-Only",
    #     "Static linear prediction",
    # ]
    # dico = dict(zip(idf.columns, cols))
    # idf.rename(columns=dico, inplace=True)

    # columns names
    idf = inddata.copy()
    cols = ["_".join(_.split("_")[2:]).lower() for _ in idf.columns]
    idf.columns = cols
    idf = idf.loc[-120:200]
    traces = cols[:4]
    se_errors = ["seup", "sedw"]
    conf_intervals = ["cirmin", "cirmax"]

    colors = [stdcolors[st] for st in ["k", "red", "red", "dark_green"]]
    #    alphas = [0.5, 0.5, 0.8, 0.8]
    alphas = [0.8, 1, 1, 1]  # changed for homogeneity
    lines = ["-", "-", "--", "--"]

    # plotting canvas
    fig = plt.figure(figsize=(180, 10))
    axes = []
    ax = fig.add_subplot(221)
    axes.append(ax)
    ax1 = fig.add_subplot(223, sharex=ax, sharey=ax)
    axes.append(ax1)
    ax = fig.add_subplot(222)
    axes.append(ax)
    ax1 = fig.add_subplot(224, sharex=ax)
    axes.append(ax1)

    # for i, ax in enumerate(axes):
    #     ax.set_title(str(i))

    # fig.suptitle(os.path.basename(filename))
    # traces
    # ax0 =============================== indi
    ax = axes[0]
    ax.set_title("Single Cell")
    for i, trace in enumerate(traces[:2]):
        ax.plot(idf[trace], color=colors[i], alpha=alphas[i], label=trace)
        if indi_std:
            ax.fill_between(
                idf[trace].index,
                idf[trace + "_" + se_errors[0]],
                idf[trace + "_" + se_errors[1]],
                color=colors[i],
                alpha=0.3,
            )
    ax.spines["bottom"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    # ax1 =============================== indi
    ax = axes[1]
    for i, trace in enumerate(traces):
        if i == 3:  # dashed
            ax.plot(
                idf[trace],
                color=colors[i],
                alpha=alphas[i],
                label=trace,
                linestyle=lines[i],
                linewidth=1.5,
            )
        else:
            ax.plot(
                idf[trace],
                color=colors[i],
                alpha=alphas[i],
                label=trace,
                linestyle=lines[i],
            )
            if indi_std:
                ax.fill_between(
                    idf[trace].index,
                    idf[trace + "_" + se_errors[0]],
                    idf[trace + "_" + se_errors[1]],
                    color=colors[i],
                    alpha=0.3,
                )
    ax.set_xlabel("Time (ms)")

    # stims locations (drawing at the end of the function)
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ["D0", "D1", "D2", "D3", "D4", "D5"]
    box_dico = dict(zip(names, hlocs))

    ################### right population part
    lp = "minus"
    popdf = popdata.copy()
    cols = [
        "centerOnly",
        "surroundThenCenter",
        "surroundOnly",
        "sosdUp",
        "sosdDown",
        "solinearPrediction",
        "stcsdUp",
        "stcsdDown",
        "stcLinearPreediction",
    ]
    popdf.drop(columns=[_ for _ in popdf.columns if "Spk" in _], inplace=True)
    dico = dict(zip(popdf.columns, cols))
    popdf.rename(columns=dico, inplace=True)
    cols = popdf.columns
    # colors = [stdcolors[st] for st in
    #           ['k', 'red', 'dark_green', 'blue_violet', 'blue_violet',
    #            'blue_violet', 'red', 'red', 'blue_violet']]
    colors = [stdcolors[st] for st in ["k", "red", "red", "blue_violet"]]
    alphas = [0.5, 0.5, 0.8, 0.5, 0.6, 0.5, 0.2, 0.2, 0.7]
    alphas = [0.8, 1, 1, 0.8]

    # sharesY = dict(minus = False, plus = True)
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
    #                          sharey=sharesY[lp], figsize=(8.5, 8))
    # axes = axes.flatten()

    # ax2 ===============================
    ax = axes[2]
    ax.set_title("Population Average")
    cols = popdf.columns[:3]
    linewidths = (1.5, 1.5, 1.5)
    for i, col in enumerate(cols):
        ax.plot(
            popdf[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=linewidths[i],
            linestyle=lines[i],
            label=col,
        )
        # std for pop
        if pop_std:
            if i == 1:  # surround then center
                ax.fill_between(
                    popdf.index,
                    popdf["stcsdUp"],
                    popdf["stcsdDown"],
                    color=colors[1],
                    alpha=0.3,
                )
            if i == 2:  # surround only
                ax.fill_between(
                    popdf.index,
                    popdf["sosdUp"],
                    popdf["sosdDown"],
                    color=colors[2],
                    alpha=0.2,
                )
    # response point
    x = 0
    y = popdf[popdf.columns[0]].loc[0]
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")

    lims = dict(minus=(-0.1, 1.1), plus=(-0.05, 1.2))
    ax.set_ylim(lims.get(lp))
    ax.set_xlim(-200, 200)

    # ax3 ===============================
    # predictive magnification
    ax = axes[3]
    colors = [stdcolors[st] for st in ["k", "red", "red", "k", "blue_violet"]]
    linewidths = (1.5, 1.5, 1.5)
    # (first, second, stdup, stddown)
    lp_cols = dict(minus=[2, 5, 3, 4], plus=[1, 6, 7, 8])
    cols = [popdf.columns[i] for i in lp_cols[lp]]
    lines = [
        "--",
    ]
    for i, col in enumerate(cols[:2]):
        print(i, col)
        ax.plot(
            popdf[col],
            color=colors[i + 2],
            alpha=alphas[i + 2],
            label=col,
            linewidth=linewidths[i],
            linestyle="--",
        )
    ax.fill_between(
        popdf.index, popdf["sosdUp"], popdf["sosdDown"], color=colors[2], alpha=0.2
    )

    for i, ax in enumerate(axes):
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
        ax.axhline(0, alpha=0.3, color="k")
        # left
        if i < 2:
            #    for ax in axes[:2]:
            ax.axvline(0, alpha=0.2, color="k")
            # response start
            x = 41
            y = idf["Center-Only"].loc[x]
            ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
            ax.vlines(x, -1, 2, color="tab:blue", linestyle=":", alpha=0.8)
            for dloc in hlocs:
                ax.axvline(dloc, linestyle=":", alpha=0.2, color="k")
            # ticks
            custom_ticks = np.linspace(0, 4, 5, dtype=int)
            ax.set_yticks(custom_ticks)
            ax.set_yticklabels(custom_ticks)

        else:
            ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
            ax.annotate(
                "n=12",
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                ha="center",
                size="large",
            )

    axes[2].xaxis.set_visible(False)
    axes[2].spines["bottom"].set_visible(False)
    axes[2].set_ylim(-0.2, 1.1)
    custom_ticks = np.arange(0, 1.1, 0.2)
    axes[2].set_yticks(custom_ticks)
    axes[3].set_xlabel("Relative Time (ms)")

    # align zero lines
    # gfunc.change_plot_trace_amplitude(ax, 0.9)
    gfunc.align_yaxis(axes[2], 0, axes[0], 0)
    fig.tight_layout()

    # left axis_label
    txt = "Membrane Potential (mV)"
    fig.text(0.03, 0.5, txt, ha="right", va="center", rotation="vertical", size="large")
    txt = "Normalized Vm"
    fig.text(0.51, 0.5, txt, ha="right", va="center", rotation="vertical", size="large")
    fig.subplots_adjust(left=0.045)

    # stimulation boxes
    lines = ["dashed", "solid"]
    vlocs = np.linspace(4.1, 3.1, 4)
    ax = axes[1]
    for name, loc in box_dico.items():
        # names
        ax.annotate(
            name,
            xy=(loc + 6, vlocs[0]),
            alpha=0.6,
            annotation_clip=False,
            fontsize="small",
        )
        # stim1 (no-center)
        # for name in box_dico.names():
        rect = Rectangle(
            xy=(loc, vlocs[3]),
            width=step,
            height=0.3,
            fill=True,
            alpha=0.6,
            edgecolor="w",
            facecolor=colors[2],
            linestyle=lines[0],
        )
        if name == "D0":
            rect = Rectangle(
                xy=(loc, vlocs[3] + 0.02),
                width=step,
                height=0.26,
                fill=True,
                alpha=1,
                linestyle=lines[0],
                edgecolor=colors[2],
                facecolor="w",
            )
        ax.add_patch(rect)
        # stim2
        rect = Rectangle(
            xy=(loc, vlocs[2]),
            width=step,
            height=0.3,
            fill=True,
            alpha=0.7,
            edgecolor="w",
            facecolor=colors[1],
        )
        ax.add_patch(rect)

        # center with all locations
        rect = Rectangle(
            xy=(loc, vlocs[1] + 0.02),
            width=step - 0.5,
            height=0.26,
            fill=False,
            alpha=0.6,
            edgecolor="k",
            facecolor=colors[0],
        )
        if name == "D0":
            rect = Rectangle(
                xy=(loc, vlocs[1]),
                width=step,
                height=0.3,
                fill=True,
                alpha=0.6,
                edgecolor="w",
                facecolor=colors[0],
            )
        ax.add_patch(rect)

    if legend:
        for ax in fig.get_axes():
            ax.legend()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fig6_predictors", ha="right", va="bottom", alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        fig.text(0.5, 0.01, "predict", ha="left", va="bottom", alpha=0.4)
    return fig


plt.close("all")
# indi_df, pop_df = build_fig6_predictors_datafile(write=False)
indi_df, pop_df = load_fig6_predictors_datafile()

fig = plot_fig6_predictors(
    inddata=indi_df, popdata=pop_df, stdcolors=std_colors, anot=anot
)
save = False
if save:
    file = "fig6_predictors"
    # to update current
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)
