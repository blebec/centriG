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


def load_indifill_datafile(key="indi", display=True):
    """ load the indifilldf dataframe (for fig 6) """
    loadfile = "example_fillin.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, loadfile)
    indifilldf = pd.read_hdf(loadfilename, "indi")
    print("-" * 20)
    print("loaded filling_in example ({})".format(key))
    if display:
        print("=" * 20, "{}({})".format(loadfile, "fillsig"))
        for column in sorted(indifilldf.columns):
            print(column)
        print()
    return indifilldf


def load_pop_datafile(key="fillsig", display=True):
    """ load the popfilldf dataframe (for fig 6 & 9) """
    try:
        key in ["pop", "pop2sig", "pop3sig", "fillsig"]
    except NameError:
        print("key shoud be in ['pop', 'pop2sig', 'pop3sig', 'fillsig']")
        return pd.DataFrame()
    loadfile = "populations_traces.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, loadfile)
    popfilldf = pd.read_hdf(loadfilename, "fillsig")
    print("-" * 20)
    print("loaded filling_in population ('{}')".format(key))
    if display:
        print("=" * 20, "{}({})".format(loadfile, "fillsig"))
        for column in sorted(popfilldf.columns):
            print(column)
        print()
    return popfilldf


popfill_df = load_pop_datafile(key="fillsig", display=False)
indifill_df = load_indifill_datafile(key="indi", display=False)

#%%


def plot_fig6_predictors(indifilldata, popfilldata, stdcolors=std_colors, anot=True):
    """
    filling in example + pop
    """
    # boolean to add std
    indi_std = True  # not used at the moment (no data available)
    pop_std = True
    # ## to build
    # indifilldata = indifill_df
    # popfilldata = popfill_df
    # stdcolors=std_colors
    # anot=True
    # ##
    legend = False
    # idf = indifilldata.copy()
    # cols = [
    #     "Center-Only",
    #     "Surround-then-Center",
    #     "Surround-Only",
    #     "Static linear prediction",
    # ]
    # dico = dict(zip(idf.columns, cols))
    # idf.rename(columns=dico, inplace=True)

    # columns names
    idf = indifilldata.copy()
    cols = ["_".join(_.split("_")[2:]).lower() for _ in idf.columns]
    idf.columns = cols
    idf = idf.loc[-120:200]
    traces = cols[:4]
    se_errors = ["seup", "sedw"]
    conf_intervals = ["cimin", "cimax"]

    colors = [stdcolors[st] for st in ["k", "red", "red", "dark_green"]]
    alphas = [1,] * 5
    alphas.insert(0, 0.8)  # black curve
    alphafill = 0.3
    linewidths = 1.5
    lines = ["-", "-", "--", "--"]

    # plotting canvas
    # fig = plt.figure(figsize=gfunc.to_inches(size))
    size = (12, 8.2)
    fig = plt.figure(figsize=size)
    axes = []
    ax = fig.add_subplot(221)
    axes.append(ax)
    ax1 = fig.add_subplot(223, sharex=ax, sharey=ax)
    axes.append(ax1)
    ax = fig.add_subplot(222)
    axes.append(ax)
    ax1 = fig.add_subplot(224, sharex=ax)
    axes.append(ax1)

    # fig.set_dpi(600)
    # fig.set_size_inches(gfunc.to_inches(size))

    # ax0 =============================== indi
    ax = axes[0]
    ax.set_title("Single Cell")
    for i, trace in enumerate(traces[:2]):
        label = "{} & ci".format(trace)
        ax.plot(idf[trace], color=colors[i], alpha=alphas[i], label=label)
        if indi_std:
            tracesup = idf["_".join([trace, conf_intervals[0]])]
            traceinf = idf["_".join([trace, conf_intervals[1]])]
            # tracesup = idf[trace] + idf["_".join([trace, conf_intervals[0]])]
            # traceinf = idf[trace] + idf["_".join([trace, conf_intervals[1]])]
            ax.fill_between(
                idf[trace].index, tracesup, traceinf, color=colors[i], alpha=0.3,
            )
    ax.spines["bottom"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if legend:
        ax.legend()

    # ax1 =============================== indi
    ax = axes[1]
    for i, trace in enumerate(traces):
        ser = idf[trace]
        if i == 3:  # dashed
            ax.plot(
                ser,
                color=colors[i],
                alpha=alphas[i],
                label=trace + " & ci",
                linestyle=lines[i],
                linewidth=linewidths,
            )
            ax.fill_between(
                ser.index,
                idf[trace + "_" + conf_intervals[0]],
                idf[trace + "_" + conf_intervals[1]],
                color=colors[i],
                alpha=alphafill,
            )
        else:
            ax.plot(
                ser,
                color=colors[i],
                alpha=alphas[i],
                label=trace + " ± se",
                linestyle=lines[i],
            )
            ax.fill_between(
                ser.index,
                idf[trace + "_" + se_errors[0]],
                idf[trace + "_" + se_errors[1]],
                color=colors[i],
                alpha=0.3,
            )
    ax.set_xlabel("Time (ms)")
    if legend:
        ax.legend()

    # stims locations (drawing at the end of the function)
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ["D0", "D1", "D2", "D3", "D4", "D5"]
    box_dico = dict(zip(names, hlocs))

    ################### right population part
    lp = "minus"  # linear predictor measure
    popdf = popfilldata.copy()
    # remove spikes and format
    popdf = popdf.drop(columns=[_ for _ in popdf.columns if "_spk_" in _])
    popdf.columns = [_.replace("popfill_vm_", "") for _ in popdf.columns]
    # cols = [
    #     "centerOnly",
    #     "surroundThenCenter",
    #     "surroundOnly",
    #     "sosdUp",
    #     "sosdDown",
    #     "solinearPrediction",
    #     "stcsdUp",
    #     "stcsdDown",
    #     "stcLinearPrediction",
    # ]
    # popdf.drop(columns=[_ for _ in popdf.columns if "Spk" in _], inplace=True)
    # dico = dict(zip(popdf.columns, cols))
    # popdf.rename(columns=dico, inplace=True)

    cols = popdf.columns
    # colors = [stdcolors[st] for st in
    #           ['k', 'red', 'dark_green', 'blue_violet', 'blue_violet',
    #            'blue_violet', 'red', 'red', 'blue_violet']]
    colors = [stdcolors[st] for st in ["k", "red", "red", "blue_violet"]]
    alphas = [0.5, 0.5, 0.8, 0.5, 0.6, 0.5, 0.2, 0.2, 0.7]
    alphas = [0.8, 1, 1, 0.8]

    # se_errors = ["seup", "sedw"]
    # conf_intervals = ["cimin", "cimax"]
    # ax2 ===============================
    ax = axes[2]
    ax.set_title("Population Average")
    linewidths = (1.5, 1.5, 1.5)
    for i, col in enumerate(cols[:3]):
        ax.plot(
            popdf[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=linewidths[i],
            linestyle=lines[i],
            label=col + " ± se",
        )
        # std for pop
        if pop_std:
            ax.fill_between(
                popdf.index,
                popdf[col + "_" + se_errors[0]],
                popdf[col + "_" + se_errors[1]],
                color=colors[i],
                alpha=alphafill,
            )
    # response point
    x = 0
    y = popdf[popdf.columns[0]].loc[0]
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")

    lims = dict(minus=(-0.1, 1.1), plus=(-0.05, 1.2))
    ax.set_ylim(lims.get(lp))
    ax.set_xlim(-200, 200)
    if legend:
        ax.legend()

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
    cols = ["s_cpiso_so", "s_cpiso_m_lp"]
    for i, col in enumerate(cols):
        print(i, col)
        ax.plot(
            popdf[col],
            color=colors[i + 2],
            alpha=alphas[i + 2],
            label=col + " ± se",
            linewidth=linewidths[i],
            linestyle="--",
        )
        if i == 0:
            ax.fill_between(
                popdf.index,
                popdf[col + "_" + se_errors[0]],
                popdf[col + "_" + se_errors[1]],
                color=colors[2],
                alpha=alphafill,
            )
        if legend:
            ax.legend()

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
            y = idf["ctr"].loc[x]
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
    gfunc.align_yaxis(axes[1], 0, axes[3], 0)

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
anot = True
fig = plot_fig6_predictors(
    indifilldata=indifill_df, popfilldata=popfill_df, stdcolors=std_colors, anot=anot
)
save = False
if save:
    # fig.set_size_inches(gfunc.to_inches(size))
    # fig.set_dpi(600)
    file = "f6_predictors"
    # to update current
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "figSup")
    for ext in [".pdf"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)
