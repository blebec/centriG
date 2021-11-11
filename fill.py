#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12 nov 2020 15:07:38 CET
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

# load data
indi_df = ldat.load_filldata("indi")
pop_df = ldat.load_filldata("pop")

#%% save the data
def print_content():
    """ print the content of the loaded data"""
    indi_content = ["_".join(_.split("Iso")) for _ in indi_df.columns]
    print("individual content = {}".format(indi_content))

    cols = [_.split("Vm")[1] for _ in pop_df.columns if "Spk" not in _]
    for re in ["Iso", "Cross", "rnd", "cp", "Se", "cf"]:
        cols = [_.replace(re, "_" + re + "_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]

    print("pop content (spk and vm=")
    for col in cols:
        print(col)


print_content()


def saveData(indidf, popdf, do_save=False):
    """save the data used to build the figure to an hdf file"""
    conds, key_dico = config.std_names()
    # key_dico = {
    #     "_s_": "_sect_",
    #     "_f_": "_full_",
    #     "_cp_": "_centripetal_",
    #     "_cf_": "_centrifugal_",
    #     "_rnd_": "_rnd_",
    #     "_Stc_": "_SthenCenter_",
    #     "_So_": "_Sonly_",
    #     "_Slp_": "_SlinearPredictor_",
    # }
    # individual
    df0 = indidf.copy()
    cols = df0.columns
    cols = ["_" + _ + "_" for _ in cols]
    # conds = [
    #     ("_s", "_s_"),
    #     ("_f", "_f_"),
    #     ("_cp", "_cp_"),
    #     ("_cf", "_cf_"),
    #     ("rnd", "_rnd_"),
    #     ("_Iso", "_iso_"),
    #     ("_Cross", "_cross_"),
    #     ("_So", "_So_"),
    #     ("_Stc", "_Stc_"),
    #     ("__", "_"),
    # ]
    for k, v in conds:
        cols = [_.replace(k, v) for _ in cols]
    for k, v in key_dico.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = [_.strip("_") for _ in cols]
    df0.columns = cols

    # pop
    df1 = popdf.drop(columns=[_ for _ in pop_df if "Spk" in _])
    cols = df1.columns
    cols = [_.split("Vm")[1] for _ in cols]
    cols = ["_" + _ + "_" for _ in cols]
    for k, v in conds:
        cols = [_.replace(k, v) for _ in cols]
    for k, v in key_dico.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = [_.strip("_") for _ in cols]
    df1.columns = cols

    data_savename = os.path.join(paths["figdata"], "fig6.hdf")
    if do_save:
        for key, df in zip(["ind", "pop"], [df0, df1]):
            df.to_hdf(data_savename, key)
    # pdframes = {}
    # for key in ['ind', 'pop']:
    #     pdframes[key] = pd.read_hdf(data_savename, key=key)


save = False
saveData(indi_df, pop_df, save)

#%%
def plot_indFill(data, stdcolors=std_colors, anot=True):
    """
    filling in example
    """
    df = data.copy()
    cols = [
        "Center-Only",
        "Surround-then-Center",
        "Surround-Only",
        "Static linear prediction",
    ]
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = [stdcolors[st] for st in ["k", "red", "dark_green", "dark_green"]]
    alphas = [0.5, 0.5, 0.8, 0.8]

    # plotting
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8.5, 8)
    )
    axes = axes.flatten()
    # fig.suptitle(os.path.basename(filename))
    # traces
    ax = axes[0]
    for i, col in enumerate(cols[:2]):
        ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i], label=col)
    ax.spines["bottom"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    ax = axes[1]
    for i, col in enumerate(cols):
        if i == 3:
            ax.plot(
                df.loc[-120:200, [col]],
                color=colors[i],
                alpha=alphas[i],
                label=col,
                linestyle="--",
                linewidth=1.5,
            )
        else:
            ax.plot(
                df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i], label=col
            )
    ax.set_xlabel("Time (ms)")
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ["D0", "D1", "D2", "D3", "D4", "D5"]
    dico = dict(zip(names, hlocs))

    ax = axes[0]
    # stimulation legend location
    uplocation = True
    if uplocation:  # location of the stimulation above the traces
        vlocs = np.linspace(2, 1, 4)
        vlocs = np.linspace(4, 3, 4)
    else:
        vlocs = np.linspace(-1.4, -2.4, 4)
        st = "Surround-then-Center"
        ax.annotate(
            st,
            xy=(30, vlocs[2]),
            color=colors[1],
            alpha=1,
            annotation_clip=False,
            fontsize="small",
        )
        st = "Center-Only"
        ax.annotate(
            st,
            xy=(30, vlocs[1]),
            color=colors[0],
            alpha=1,
            annotation_clip=False,
            fontsize="small",
        )
        # see annotation_clip=False
        ax.set_ylim(-2.5, 4.5)

        for key in dico.keys():
            # name
            ax.annotate(
                key,
                xy=(dico[key] + 6, vlocs[0]),
                alpha=0.6,
                annotation_clip=False,
                fontsize="small",
            )
            # stim1
            rect = Rectangle(
                xy=(dico[key], vlocs[2]),
                width=step,
                height=0.3,
                fill=True,
                alpha=0.6,
                edgecolor="w",
                facecolor=colors[1],
            )
            ax.add_patch(rect)
        # center
        rect = Rectangle(
            xy=(0, vlocs[1]),
            width=step,
            height=0.3,
            fill=True,
            alpha=0.6,
            edgecolor="w",
            facecolor=colors[0],
        )  #'k')
        ax.add_patch(rect)

    # ax2
    ax = axes[1]
    for key in dico.keys():
        # names
        ax.annotate(
            key,
            xy=(dico[key] + 6, vlocs[0]),
            alpha=0.6,
            annotation_clip=False,
            fontsize="small",
        )
        # stim1
        rect = Rectangle(
            xy=(dico[key], vlocs[3]),
            width=step,
            height=0.3,
            fill=True,
            alpha=1,
            edgecolor="w",
            facecolor=colors[2],
        )
        if key == "D0":
            rect = Rectangle(
                xy=(dico[key], vlocs[3] + 0.02),
                width=step,
                height=0.26,
                fill=True,
                alpha=1,
                edgecolor=colors[2],
                facecolor="w",
            )
        ax.add_patch(rect)
        # stim2
        rect = Rectangle(
            xy=(dico[key], vlocs[2]),
            width=step,
            height=0.3,
            fill=True,
            alpha=0.6,
            edgecolor="w",
            facecolor=colors[1],
        )
        ax.add_patch(rect)

        # center with all locations
        rect = Rectangle(
            xy=(dico[key], vlocs[1] + 0.02),
            width=step - 0.5,
            height=0.26,
            fill=False,
            alpha=0.6,
            edgecolor="k",
            facecolor=colors[0],
        )
        if key == "D0":
            rect = Rectangle(
                xy=(dico[key], vlocs[1]),
                width=step,
                height=0.3,
                fill=True,
                alpha=0.6,
                edgecolor="w",
                facecolor=colors[0],
            )
        ax.add_patch(rect)

    # #center (without the missing stims)
    # rect = Rectangle(xy=(0, vlocs[1]), width=step, height=0.3, fill=True,
    #                  alpha=0.6, edgecolor='w', facecolor=colors[0])

    ax.add_patch(rect)
    if not uplocation:
        for i, st in enumerate(
            ["Center-Only", "Surround-then-Center", "Surround-Only"]
        ):
            ax.annotate(
                st,
                xy=(30, vlocs[i + 1]),
                color=colors[i],
                annotation_clip=False,
                fontsize="small",
            )
    txt = "Membrane Potential (mV)"
    fig.text(0.03, 0.5, txt, ha="right", va="center", rotation="vertical")
    for ax in axes:
        # leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
        #                handlelength=0)
        # colored text
        # for line, text in zip(leg.get_lines(), leg.get_texts()):
        # text.set_color(line.get_color())
        #        ax.set_ylabel('Membrane potential (mV)')
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
        ax.axhline(0, alpha=0.2, color="k")
        ax.axvline(0, alpha=0.2, color="k")
        # response start
        x = 41
        y = df["Center-Only"].loc[x]
        ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
        ax.vlines(x, -1, 2, color="tab:blue", linestyle=":", alpha=0.8)
        for dloc in hlocs:
            ax.axvline(dloc, linestyle=":", alpha=0.2, color="k")
        # ticks
        custom_ticks = np.linspace(0, 4, 5, dtype=int)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.99, 0.01, "fill.py:plot_indFill", ha="right", va="bottom", alpha=0.4)
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


plt.close("all")
# indi_df = load_filldata('indi')
fig = plot_indFill(indi_df, std_colors, anot=anot)
save = False
if save:
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    file_name = os.path.join(dirname, "indFill.png")
    fig.savefig(file_name)

#%%
plt.close("all")

fig = figp.plot_indFill_bis(indi_df, std_colors)
# fig = plot_figure6_bis(substract=True)
fig = figp.plot_indFill_bis(indi_df, std_colors, linear=False, substract=True)

#%%


def plot_pop_predict(data, lp="minus", stdcolors=std_colors):
    """
    plot_figure7
    lP <-> linear predictor
    """
    df = data.copy()
    cols = [
        "centerOnly",
        "surroundThenCenter",
        "surroundOnly" "sosdUp",
        "sosdDown",
        "solinearPrediction",
        "stcsdUp",
        "stcsdDown",
        "stcLinearPreediction",
    ]
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = [
        stdcolors[st]
        for st in [
            "k",
            "red",
            "dark_green",
            "blue_violet",
            "blue_violet",
            "blue_violet",
            "red",
            "red",
            "blue_violet",
        ]
    ]
    alphas = [0.5, 0.5, 0.8, 0.5, 0.6, 0.5, 0.2, 0.2, 0.7]

    sharesY = dict(minus=False, plus=True)
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=sharesY[lp], figsize=(8.5, 8)
    )
    axes = axes.flatten()

    ax = axes[0]
    cols = df.columns[:3]
    linewidths = (1.5, 1.5, 1.5)
    for i, col in enumerate(cols):
        ax.plot(
            df[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=linewidths[i],
            label=col,
        )
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")

    lims = dict(minus=(-0.1, 1.1), plus=(-0.05, 1.2))
    ax.set_ylim(lims.get(lp))
    ax.set_xlim(-200, 200)

    # predictive magnification
    ax = axes[1]
    colors = [stdcolors[st] for st in ["k", "red", "dark_green", "blue_violet"]]
    linewidths = (1.5, 1.5, 1.5)
    # (first, second, stdup, stddown)
    lp_cols = dict(minus=[2, 5, 3, 4], plus=[1, 6, 7, 8])
    cols = [df.columns[i] for i in lp_cols[lp]]
    for i, col in enumerate(cols[:2]):
        ax.plot(
            df[col],
            color=colors[i + 2],
            alpha=alphas[i + 2],
            label=col,
            linewidth=linewidths[i],
        )
    ax.fill_between(df.index, df[cols[2]], df[cols[3]], color=colors[2], alpha=0.1)

    for i, ax in enumerate(axes):
        ax.axhline(0, alpha=0.3, color="k")
        ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
        ax.set_ylabel("Normalized membrane potential")
        ax.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
        if i > 0:
            ax.set_xlabel("Relative time (ms)")
            # if lp == 'minus':
            #     gfunc.change_plot_trace_amplitude(ax, 0.9)
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if lp == "minus":
        custom_ticks = np.arange(0, 1.1, 0.2)
        axes[0].set_yticks(custom_ticks)
    elif lp == "plus":
        custom_ticks = np.arange(0, 1.2, 0.2)
        axes[0].set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fill.py.py:pop_predict", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


plt.close("all")

fig = plot_pop_predict(pop_df, "minus")
# fig = plot_pop_predict('plus')
fig2 = figp.plot_pop_fill_bis(pop_df, std_colors)
save = False
if save:
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    file_name = os.path.join(dirname, "predict_Fill.pdf")
    fig.savefig(file_name)

#%% pop + predict


def plot_indFill_popPredict(inddata, popdata, stdcolors=std_colors, anot=True):
    """
    filling in example + pop
    """

    # ## to build
    # inddata = indi_df
    # popdata = pop_df
    # stdcolors=std_colors
    # anot=True
    # ##
    legend = False
    idf = inddata.copy()
    cols = [
        "Center-Only",
        "Surround-then-Center",
        "Surround-Only",
        "Static linear prediction",
    ]
    dico = dict(zip(idf.columns, cols))
    idf.rename(columns=dico, inplace=True)
    # color parameters
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
    # ax0 ===============================
    ax = axes[0]
    ax.set_title("Single Cell")
    for i, col in enumerate(cols[:2]):
        ax.plot(idf.loc[-120:200, [col]], color=colors[i], alpha=alphas[i], label=col)
    ax.spines["bottom"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    # ax1 ===============================
    ax = axes[1]
    for i, col in enumerate(cols):
        if i == 3:  # dashed
            ax.plot(
                idf.loc[-120:200, [col]],
                color=colors[i],
                alpha=alphas[i],
                label=col,
                linestyle=lines[i],
                linewidth=1.5,
            )
        else:
            ax.plot(
                idf.loc[-120:200, [col]],
                color=colors[i],
                alpha=alphas[i],
                label=col,
                linestyle=lines[i],
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
        ax.plot(
            popdf[col],
            color=colors[i + 2],
            alpha=alphas[i + 2],
            label=col,
            linewidth=linewidths[i],
            linestyle="--",
        )
    ax.fill_between(
        popdf.index, popdf[cols[2]], popdf[cols[3]], color=colors[2], alpha=0.2
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
    for key in box_dico.keys():
        # names
        ax.annotate(
            key,
            xy=(box_dico[key] + 6, vlocs[0]),
            alpha=0.6,
            annotation_clip=False,
            fontsize="small",
        )
        # stim1 (no-center)
        # for key in box_dico.keys():
        rect = Rectangle(
            xy=(box_dico[key], vlocs[3]),
            width=step,
            height=0.3,
            fill=True,
            alpha=0.6,
            edgecolor="w",
            facecolor=colors[2],
            linestyle=lines[0],
        )
        if key == "D0":
            rect = Rectangle(
                xy=(box_dico[key], vlocs[3] + 0.02),
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
            xy=(box_dico[key], vlocs[2]),
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
            xy=(box_dico[key], vlocs[1] + 0.02),
            width=step - 0.5,
            height=0.26,
            fill=False,
            alpha=0.6,
            edgecolor="k",
            facecolor=colors[0],
        )
        if key == "D0":
            rect = Rectangle(
                xy=(box_dico[key], vlocs[1]),
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
            0.99,
            0.01,
            "fill.py:plot_indFill_popPredict",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        fig.text(0.5, 0.01, "predict", ha="left", va="bottom", alpha=0.4)
    return fig


plt.close("all")
# indi_df = load_filldata('indi')
fig = plot_indFill_popPredict(
    inddata=indi_df, popdata=pop_df, stdcolors=std_colors, anot=anot
)
save = False
if save:
    file = "indFill_popPredict"
    ext = ".png"
    folder = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    filename = os.path.join(folder, (file + ext))
    fig.savefig(filename)
    # to update current
    file = "f11_" + file
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)


#%%


def plot_pop_fill(data, stdcolors=std_colors, anot=anot):
    """
    plot_figure7
    lP <-> linear predictor
    """
    df = data.copy()
    cols = gfunc.new_columns_names(df.columns)
    cols = ["_".join(item.split("_")[1:]) for item in cols]
    df.columns = cols

    cols = [
        "centerOnly",
        "surroundThenCenter",
        "surroundOnly",
        "sosdUp",
        "sosdDown",
        "solinearPrediction",
        "stcsdUp",
        "stcsdDown",
        "stcLinearPrediction",
        "stcvmcfIso",
        "stcvmcpCross",
        "stcvmfRnd",
        "stcvmsRnd",
        "stcspkcpCtr, stcspkcpIso",
        "stcspkcfIso",
        "stcspkcpCross",
        "stcspkfRnd",
        "stcspksRnd",
    ]
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = [stdcolors[st] for st in ["k", "red", "green", "yellow", "blue", "blue"]]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]

    spks = cols[13:17]
    vms = [df.columns[i] for i in [0, 1, 9, 10, 11]]

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(5.5, 13), sharex=True, sharey=True
    )
    axes = axes.flatten()

    # vm pop
    ax = axes[0]
    for i, col in enumerate(vms):
        ax.plot(
            df[col], color=colors[i], alpha=alphas[i], linewidth=2, label=df.columns[i]
        )
    ax.set_xlim(-20, 50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.set_ylabel("Normalized membrane potential")
    ax.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
    # spk pop
    ax = axes[1]
    for i, col in enumerate(spks):
        ax.plot(
            df[col], color=colors[i], alpha=alphas[i], linewidth=2, label=df.columns[i]
        )
    x = 0
    y = df[spks[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.set_xlim(-20, 60)
    ax.set_ylabel("Normalized firing rate")
    ax.annotate("n=7", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
    ax.set_xlabel("Relative time (ms)")

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.axhline(0, alpha=0.3, color="k")
        ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
        custom_ticks = np.arange(0, 1.1, 0.2)
        ax.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fill.py:plot_pop_fill", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")

fig = plot_pop_fill(pop_df, std_colors, anot)
save = False
if save:
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    file = os.path.join(dirname, "pop_fill.png")
    fig.savefig(os.path.join(dirname, file))


#%%


def plot_pop_fill_2X2(df, lp="minus", stdcolors=std_colors):
    """
    plot_figure7
    lP <-> linear predictor
    """

    cols = gfunc.new_columns_names(df.columns)
    cols = ["_".join(item.split("_")[1:]) for item in cols]
    # df.columns = cols

    # cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
    #         'sosdUp', 'sosdDown', 'solinearPrediction', 'stcsdUp',
    #         'stcsdDown', 'stcLinearPrediction',
    #         'stcvmcfIso', 'stcvmcpCross', 'stcvmfRnd', 'stcvmsRnd',
    #         'stcspkcpCtr, stcspkcpIso',
    #         'stcspkcfIso', 'stcspkcpCross','stcspkfRnd', 'stcspksRnd']
    # dico = dict(zip(df.columns, cols))
    # df.rename(columns=dico, inplace=True)
    # colors = ['k', std_colors['red'], std_colors['dark_green'],
    #           std_colors['blue_violet'], std_colors['blue_violet'],
    #           std_colors['blue_violet'], std_colors['red'],
    #           std_colors['red'], std_colors['blue_violet'],
    #           std_colors['green'], std_colors['yellow'],
    #           std_colors['blue'], std_colors['blue'],
    #           'k', std_colors['red'],
    #           std_colors['green'], std_colors['yellow'],
    #           std_colors['blue'], std_colors['blue']]
    colors = [
        stdcolors[st]
        for st in [
            "k",
            "red",
            "dark_green",
            "blue_violet",
            "blue_violet",
            "blue_violet",
            "red",
            "red",
            "blue_violet",
            "green",
            "yellow",
            "blue",
            "blue",
            "k",
            "red",
            "green",
            "yellow",
            "blue",
            "blue",
        ]
    ]
    alphas = [
        0.5,
        0.7,
        0.7,
        0.5,
        0.5,
        0.6,
        0.2,
        0.2,
        0.7,
        0.7,
        0.7,
        0.7,
        0.7,
        0.5,
        0.7,
        0.7,
        0.7,
        0.7,
        0.7,
    ]

    fig = plt.figure(figsize=(11.6, 10))
    # fig.suptitle(os.path.basename(filename))

    # traces & surround only
    ax0 = fig.add_subplot(221)
    cols = df.columns[:3]
    linewidths = (1, 1, 2)
    for i, col in enumerate(cols):
        ax0.plot(
            df[col],
            color=colors[i],
            alpha=alphas[i],
            linewidth=linewidths[i],
            label=col,
        )
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax0.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    lims = dict(minus=(-0.2, 1.1), plus=(-0.05, 1.2))
    ax0.set_ylim(lims.get(lp))

    # surroundOnly & predictor
    if lp == "minus":
        ax1 = fig.add_subplot(223, sharex=ax0)
    elif lp == "plus":
        ax1 = fig.add_subplot(223, sharex=ax0, sharey=ax0)
    colors = [stdcolors[st] for st in ["k", "red", "dark_green", "blue_violet"]]
    linewidths = (2, 1)
    # (first, second, stdup, stddown)
    lp_cols = dict(minus=[2, 5, 3, 4], plus=[1, 6, 7, 8])
    cols = [df.columns[i] for i in lp_cols[lp]]
    for i, col in enumerate(cols[:2]):
        ax1.plot(
            df[col],
            color=colors[i + 2],
            alpha=alphas[i + 2],
            label=col,
            linewidth=linewidths[i],
        )
    ax1.fill_between(df.index, df[cols[2]], df[cols[3]], color=colors[2], alpha=0.1)

    # populations
    colors = [stdcolors[st] for st in ["k", "red", "green", "yellow", "blue", "blue"]]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]
    spk_cols = df.columns[13:18]
    vm_cols = [df.columns[i] for i in [0, 1, 9, 10, 11]]
    # vm
    ax2 = fig.add_subplot(222, sharey=ax0)
    for i, col in enumerate(vm_cols):
        ax2.plot(
            df[col], color=colors[i], alpha=alphas[i], linewidth=2, label=df.columns[i]
        )
    ax2.set_xlim(-20, 50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax2.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax2.set_ylabel("Normalized membrane potential")
    ax2.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")

    # spk
    ax3 = fig.add_subplot(224, sharex=ax2)
    for i, col in enumerate(spk_cols):
        ax3.plot(
            df[col], color=colors[i], alpha=alphas[i], linewidth=2, label=df.columns[i]
        )
    x = 0
    y = df[spk_cols[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = 0.06  # vertical spread for realign location
    ax3.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax3.set_ylabel("Normalized firing rate")
    ax3.annotate("n=7", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
    ax3.set_xlabel("Relative time (ms)")

    ax0.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
    ax0.set_ylabel("Normalized membrane potential")
    ax2.set_ylabel("Normalized membrane potential")
    ax1.set_xlabel("Relative time (ms)")
    ax3.set_xlabel("Relative time (ms)")
    ax3.set_ylabel("Normalized firing rate")
    ax3.annotate("n=7", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")

    for ax in fig.get_axes():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
        ax.axhline(0, alpha=0.3, color="k")
        ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
        # ax.axvline(0, alpha=0.3, color='k')
    # align zero between subplots
    # gfunc.align_yaxis(ax1, 0, ax2, 0)
    if lp == "minus":
        gfunc.change_plot_trace_amplitude(ax2, 0.9)
    fig.tight_layout()
    # add ref
    ref = (0, df.loc[0, [df.columns[0]]])

    if lp == "minus":
        custom_ticks = np.arange(0, 1.1, 0.2)
        ax0.set_yticks(custom_ticks)
    elif lp == "plus":
        custom_ticks = np.arange(0, 1.2, 0.2)
        ax1.set_yticks(custom_ticks)
        ax3.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fill.py:plot_pop_fill_2X2", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


plt.close("all")
# fig1 = plot_pop_fill_2X2(pop_df, 'plus', std_colors)
fig2 = plot_pop_fill_2X2(pop_df, "minus", std_colors)
save = False
if save:
    # file = 'pop_fill_2X2_plus.png'
    # fig1.savefig(os.path.join(paths['save'], file))
    file = "pop_fill_2X2_minus.png"
    fig2.savefig(os.path.join(paths["save"], file))

#%%


def plot_pop_fill_surround(data, stdcolors=std_colors):
    """
    plot_figure7 surround only vm responses

    """
    df = data.copy()
    cols = gfunc.new_columns_names(df.columns)
    cols = ["_".join(item.split("_")[1:]) for item in cols]
    df.columns = cols

    # fig = plt.figure(figsize=(11.6, 10))
    fig = plt.figure(figsize=(6.5, 5.5))
    # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    surround_cols = [df.columns[st] for st in (2, 19, 20, 21)]
    colors = [stdcolors[st] for st in ["k", "red", "green", "yellow", "blue", "blue"]]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]
    # +1 because no black curve
    for i, col in enumerate(surround_cols):
        # for i in (2,19,20,21):
        ax.plot(
            df[col], color=colors[i + 1], alpha=alphas[i + 1], linewidth=2, label=col
        )
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    # vspread = .06  # vertical spread for realign location
    # ax1.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-150, 150)

    ax.set_ylabel("Normalized membrane potential")
    ax.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha="center")
    ax.set_xlabel("Relative time (ms)")

    for ax in fig.get_axes():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.axhline(0, alpha=0.3, color="k")
        ax.axvline(0, linewidth=2, color="tab:blue", linestyle=":")
    # align zero between subplots
    # gfunc.align_yaxis(ax1, 0, ax2, 0)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "fill.py:plot_pop_fill_surround",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


plt.close("all")

fig = plot_pop_fill_surround(pop_df, std_colors)

# # to use with pop_subtraction
# ax = fig.get_axes()[0]
# ax.set_xlim(-150, 150)
# ax.set_ylim(-0.15, 0.35)
# ax.set_xticks(np.linspace(-150, 150, 7))
# ax.set_yticks(np.linspace(-0.1, 0.3, 5))

save = False
if save:
    dirname = os.path.join(
        paths["owncFig"], "pythonPreview", "fillingIn", "indFill_popFill"
    )
    file = "pop_fill_surround.png"
    fig.savefig(os.path.join(dirname, file))

#%% plot combi
plt.close("all")


def plot_fill_combi(data_fill, data_pop, stdcolors=std_colors, anot=anot):

    colors = [stdcolors[st] for st in ["k", "red", "green", "yellow", "blue", "blue"]]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    # fill pop
    df = data_fill.copy()

    # general  pop
    gen_df = data_pop.copy()
    # defined in dataframe columns (first column = ctr))
    kind, rec, spread, *_ = gen_df.columns.to_list()[1].split("_")
    # centering
    middle = (gen_df.index.max() - gen_df.index.min()) / 2
    gen_df.index = (gen_df.index - middle) / 10
    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    # subtract the centerOnly response (ref = df['CENTER-ONLY'])
    ref = gen_df[gen_df.columns[0]]
    gen_df = gen_df.subtract(ref, axis=0)
    # remove rdsect
    cols = gen_df.columns.to_list()
    while any(st for st in cols if "sect_rd" in st):
        cols.remove(next(st for st in cols if "sect_rd" in st))
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

#%% save the data for figure 9


def save_fig9_data(do_save=False):
    # pop_df  and data_df
    conds, key_dico = config.std_names()

    select = dict(age="new", rec="vm", kind="sig")
    data_df, file = ltra.load_intra_mean_traces(paths, **select)

    data_savename = os.path.join(paths["figdata"], "fig9.hdf")
    do_save = False
    if do_save:
        data_df.to_hdf(data_savename, "popSig")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "popFill"))
    for item in data_df.columns:
        print(item)
    print()

    pop_df = ldat.load_filldata("pop")
    df1 = pop_df.copy()
    cols = df1.columns
    cols = ["_" + _ + "_" for _ in cols]
    for k, v in conds:
        cols = [_.replace(k, v) for _ in cols]
    for k, v in key_dico.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = [_.strip("_") for _ in cols]
    df1.columns = cols

    data_savename = os.path.join(paths["figdata"], "fig9.hdf")
    if do_save:
        df1.to_hdf(data_savename, "popFill")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "popFill"))
    for item in cols:
        print(item)
    print()


save_fig9_data(False)
