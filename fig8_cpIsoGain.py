#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:05:25 2021

@author: cdesbois
"""
import os
from datetime import datetime
from importlib import reload

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import config
import fig_proposal as figp
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra

# import old.old_figs as ofig

# import itertools


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
paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)
paths["figSup"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "figSup")

os.chdir(paths["pg"])


def load_f8_cpIsoGain_data(display=False):
    """ load fig8_cpsisogain hdf files
    input:
        display : boolean to list the files
    return
        indidf, popdf, pop2sigdf, pop3sigdf : pandas_Dataframes
    """
    file = "fig8s.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, file)
    indidf = pd.read_hdf(loadfilename, key="indi")
    popdf = pd.read_hdf(loadfilename, key="pop")
    pop2sigdf = pd.read_hdf(loadfilename, key="pop2sig")
    pop3sigdf = pd.read_hdf(loadfilename, key="pop3sig")

    keys = ["indi", "pop", "pop2sig", "pop3sig"]
    dfs = [indidf, popdf, pop2sigdf, pop3sigdf]
    for key, df in zip(keys, dfs):
        print("=" * 20, "loaded {}({})".format(file, key))
        if display:
            for column in sorted(df.columns):
                print(column)
        print()
    return indidf, popdf, pop2sigdf, pop3sigdf


indi_df, pop_df, pop2sig_df, pop3sig_df = load_f8_cpIsoGain_data()

#%%
plt.close("all")


def plot_fig8_cpIsoGain(
    indidf, popdf, pop2sigdf, pop3sigdf, anot=False, stdcolors=std_colors
):
    """
    figure2 (individual + pop + sig)
    input:
        datadf
        colsdict
        anot = boolean (display of anotations)
        age in ['new, 'old'] : choose the related population
        onlyPos = boolean to display only positive values for spike
    output: pyplot figure
    """
    # to build
    # indidf = indi_df
    # popdf = pop_df
    # pop2sigdf = pop2sig_df
    # pop3sigdf = pop3sig_df
    # stdcolors = std_colors

    # limit the spread
    popdf = popdf.loc[-20:60]  # limit xscale
    pop2sigdf = pop2sigdf.loc[-20:60]
    pop3sigdf = pop3sigdf.loc[-20:60]

    colors = [stdcolors[_] for _ in "k red".split()]
    alphas = (0.8, 0.8)
    vspread = 0.06  # vertical spread for realign location

    # build fig canvas
    fig = plt.figure(figsize=(18, 16))
    axes = []
    ax0 = fig.add_subplot(241)
    ax1 = fig.add_subplot(242)
    axes.append(ax0)
    axes.append(ax1)
    axes.append(fig.add_subplot(243, sharex=ax1, sharey=ax1))
    axes.append(fig.add_subplot(244, sharex=ax1, sharey=ax1))

    axes.append(fig.add_subplot(245, sharex=ax0))
    ax2 = fig.add_subplot(246, sharex=ax1)
    axes.append(ax2)
    axes.append(fig.add_subplot(247, sharex=ax1, sharey=ax2))
    axes.append(fig.add_subplot(248, sharex=ax1, sharey=ax2))
    vmaxes = axes[:4]  # vm axes = top row
    spkaxes = axes[4:]  # spikes axes = bottom row

    seerrors = ["seup", "sedw"]

    # ___ individual vm
    indidf.columns = [_.replace("indi_", "") for _ in indidf.columns]
    cols = [_ for _ in indidf.columns if "vm_" in _]
    cols = cols[:2]
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ser = indidf[col]
        ax.plot(ser, color=colors[i], alpha=alphas[i], label=col)
        ax.fill_between(
            ser.index,
            indidf[col + "_" + seerrors[0]],
            indidf[col + "_" + seerrors[1]],
            color=colors[i],
            alpha=0.3,
        )
    # response point
    x = 43.5
    y = indidf.loc[x, [indidf.columns[0]]]
    ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ individual spike
    cols = [_ for _ in indidf.columns if "spk_" in _]
    cols = cols[:2]
    ax = spkaxes[0]
    rev_cols = cols[::-1]
    rev_colors = colors[::-1]
    for i, col in enumerate(rev_cols):
        ser = indidf[col]
        ax.plot(indidf[col], color=rev_colors[i], alpha=1, label=col, linewidth=1)
        ax.fill_between(indidf.index, indidf[col], color=rev_colors[i], alpha=0.5)
    # response point
    x = 55.5
    y = indidf.loc[x, [cols[0]]]
    ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ pop vm
    cols = [_ for _ in popdf.columns if "_vm_" in _][:2]
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(popdf[col], color=colors[i], alpha=alphas[i], label=col, linewidth=1.5)
        ax.fill_between(
            popdf.index,
            popdf[col + "_" + seerrors[0]],
            popdf[col + "_" + seerrors[1]],
            color=colors[i],
            alpha=0.3,
        )
    ax.annotate(
        "n=37",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    # response point
    x = 0
    y = popdf.loc[x, [cols[0]]]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ pop spike
    cols = [_ for _ in popdf.columns if "_spk_" in _ and "_se" not in _]
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(popdf[col], color=rev_colors[i], alpha=1, label=col, linewidth=1.5)
        ax.fill_between(
            popdf.index,
            popdf[col + "_" + seerrors[0]],
            popdf[col + "_" + seerrors[1]],
            color=rev_colors[i],
            alpha=0.3,
        )
    ax.annotate(
        "n=22",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )
    # response point
    x = 0
    y = popdf.loc[x, [cols[0]]]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # popVm2Sig
    cols = [_ for _ in pop2sigdf.columns if "_vm" in _ and "_se" not in _]
    ax = vmaxes[2]
    # traces
    for i, col in enumerate(cols):
        ax.plot(pop2sigdf[col], color=colors[i], alpha=alphas[i], label=col)
        ax.fill_between(
            pop2sigdf.index,
            pop2sigdf[col + "_" + seerrors[0]],
            pop2sigdf[col + "_" + seerrors[1]],
            color=colors[i],
            alpha=0.3,
        )
    # response point
    x = 0
    y = pop2sigdf.loc[x, [cols[0]]]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")
    ax.annotate(
        "n=15",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    # popSpk2Sig
    cols = [_ for _ in pop2sigdf.columns if "_spk" in _ and "_se" not in _]
    ax = spkaxes[2]
    # traces
    for i, col in enumerate(cols[::-1]):
        ax.plot(
            pop2sigdf[col],
            color=colors[::-1][i],
            alpha=alphas[::-1][i],
            label=col,
            linewidth=2,
        )
        ax.fill_between(
            pop2sigdf.index,
            pop2sigdf[col + "_" + seerrors[0]],
            pop2sigdf[col + "_" + seerrors[1]],
            color=colors[::-1][i],
            alpha=0.3,
        )
    # response point
    x = 0
    y = pop2sigdf.loc[x, [cols[0]]]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")
    ax.annotate(
        "n=6",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    ######## 3SIG #######
    colors = [stdcolors[color] for color in "k red green yellow blue blue".split()]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    nbcells = dict(sect=[20, 10], full=[15, 7])  # [vm, spk]

    axes = [vmaxes[-1], spkaxes[-1]]
    records = ["vm"] + ["spk"]
    cols = [_ for _ in pop3sigdf.columns if "_se" not in _]
    cols = [_ for _ in cols if "_frnd" not in _]

    # plot
    for i, ax in enumerate(axes):
        record = records[i]
        txt = "n={}".format(nbcells["sect"][i])
        ax.annotate(
            txt,
            xy=(0.9, 0.9),
            size="large",
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
        traces = [_ for _ in cols if record in _]
        # replace rd sector by full sector
        traces = [_.replace("srnd", "frnd") for _ in traces]

        df = pop3sigdf[traces].copy()
        substract = False
        if substract:
            # subtract the centerOnly response (ref = df['CENTER-ONLY'])
            ref = df[df.columns[0]]
            df = df.subtract(ref, axis=0)
        # build labels
        labels = ["_".join(_.split("_")[2:]) for _ in traces]
        labels = [_.replace("_stc", "") for _ in labels]
        # plot
        for j, col in enumerate(traces):
            ax.plot(
                df[col],
                color=colors[j],
                alpha=alphas[j],
                label=labels[j],
                linewidth=1.5,
            )
            # [0, 1, 2, 3, 4]
            if j in [
                0,
                1,
            ]:
                ax.fill_between(
                    pop3sigdf.index,
                    pop3sigdf[col + "_" + seerrors[0]],
                    pop3sigdf[col + "_" + seerrors[1]],
                    color=colors[j],
                    alpha=0.3,
                )
        # bluePoint
        x = 0
        y = df.loc[0, [df.columns[0]]]
        vspread = 0.06  # vertical spread for realign location
        # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
        ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
        ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # labels
    ylabels_vm = ["Membrane Potential (mV)", "Normalized Vm", "", ""]

    ylabels_spk = ["Firing Rate (Spk/s)", "Normalized Spk/s", "", ""]
    ylabels = ylabels_vm + ylabels_spk
    axes = fig.get_axes()
    for i, ax in enumerate(axes):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            ax.set_ylabel(ylabels[i])
        if i < 4:
            ax.axes.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel("Time (ms)")
    axes[1].set_ylim(-0.10, 1.1)
    for ax in axes[5:]:
        ax.set_xlabel("Relative Time (ms)")
        ax.set_ylim(-0.10, 1)

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ["D" + str(i) for i in range(6)]
    dico = dict(zip(names, xlocs))
    # lines
    for ax in [vmaxes[0], spkaxes[0]]:
        for dloc in xlocs:
            ax.axvline(dloc, linestyle=":", alpha=0.4, color="k")
    # fit individual example
    vmaxes[0].set_ylim(-3.5, 12)
    spkaxes[0].set_ylim(-10, 35)
    # stim location
    ax = spkaxes[0]
    for key, val in dico.items():
        ax.annotate(
            key, xy=(val + step / 2, -5), alpha=0.6, ha="center", fontsize="x-small",
        )
        # stim
        rect = Rectangle(
            xy=(val, -9),
            width=step,
            height=2,
            fill=True,
            alpha=0.6,
            edgecolor="w",
            facecolor=std_colors["red"],
        )
        ax.add_patch(rect)
    # center
    rect = Rectangle(
        xy=(0, -7),
        width=step,
        height=2,
        fill=True,
        alpha=0.6,
        edgecolor="w",
        facecolor="k",
    )
    ax.add_patch(rect)

    # align zero between plots  NB ref = first plot
    gfunc.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    gfunc.align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    gfunc.change_plot_trace_amplitude(vmaxes[1], 0.85)
    gfunc.change_plot_trace_amplitude(spkaxes[1], 0.8)

    # zero line
    for ax in axes:
        ax.axhline(0, alpha=0.4, color="k")
    # scales vm
    ax = vmaxes[0]
    ax.axvline(0, alpha=0.4, color="k")
    custom_ticks = np.linspace(-2, 10, 7, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # scales spk
    ax = spkaxes[0]
    ax.axvline(0, alpha=0.4, color="k")
    custom_ticks = np.linspace(0, 30, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # scales pop
    for ax in vmaxes[1:]:
        custom_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(custom_ticks)
    for ax in spkaxes[1:]:
        custom_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(custom_ticks)

    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06, wspace=0.4)
    # align ylabels
    fig.align_ylabels()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "fig8_cpIsoGain", ha="right", va="bottom", alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        # fig.text(0.5, 0.01, 'fig2', ha='center', va='bottom', alpha=0.4)
    return fig


plt.close("all")

figure = plot_fig8_cpIsoGain(
    indidf=indi_df, popdf=pop_df, pop2sigdf=pop2sig_df, pop3sigdf=pop3sig_df, anot=anot,
)
save = False
if save:
    name = "f8_cpIsoGain"
    # paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["figSup"], (name + ext)))
