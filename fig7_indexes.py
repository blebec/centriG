#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:15:13 2021

@author: cdesbois
"""


import os
from datetime import datetime
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import general_functions as gfunc
import load.load_data as ldat

# ===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])


def load_measures(display=False):
    key = "indexes"
    data_loadname = os.path.join(paths["figdata"], "measures.hdf")
    df = pd.read_hdf(data_loadname, key=key)
    conds = {_.split("_")[0] for _ in df.columns}
    dico = {}
    for cond in conds:
        dico[cond] = ["_".join(_.split("_")[1:]) for _ in df.columns if cond in _]
    print("=" * 20, "{}(key={})".format(os.path.basename(data_loadname), key))
    for k, v in dico.items():
        print(k, v)
    print()
    return df


#%%
def plot_all_cg_sorted_responses(indexeddf=None, **kwargs):
    #     overlap=True, sort_all=True, key=0, spread="sect", rec="vm", age="new", amp="engy"
    # ):
    """plot the sorted cell responses
    input : conditions parameters
        key : number to choose the trace to be sorted
        overlap = (True) boolean, overlap the different rows to superpose the plots
        sort_all = (True) if false, only the 'key' trace is sorted
        key = (0) integer, trace to be sorted
        spread = space_spread in [(sect), full]
        rec = record_kind in [('vm'), 'spk']
        age = record age in [(new), old] (old <-> time50, gain50, old way)
        amp = amplitude in [(engy), gain]
    output :
        matplotlib.pyplot
    """
    overlap = kwargs.get("overlap", True)
    sort_all = kwargs.get("sort_all", True)
    key = kwargs.get("key", 0)
    spread = kwargs.get("spread", "sect")
    rec = kwargs.get("rec", "vm")
    age = kwargs.get("age", "new")
    amp = kwargs.get("amp", "engy")

    titles = config.std_titles()
    stdcolors = config.std_colors()

    def set_ticks_both(axis):
        """ set ticks and ticks labels on both sides """
        ticks = list(axis.majorTicks)  # a copy
        ticks.extend(axis.minorTicks)
        for t in ticks:
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
            t.label1.set_visible(True)
            t.label2.set_visible(True)

    # parameter
    colors = [stdcolors[item] for item in ["red", "green", "yellow", "blue", "blue"]]
    colors = [color for color in colors for _ in range(2)]
    # data (call)
    if indexeddf is None:
        df = ldat.load_cell_contributions(rec=rec, amp=amp, age=age)
    else:
        df = indexed_df.copy()
    # extract list of traces : sector vs full
    traces = [_ for _ in df.columns if spread in _]
    # remove the 'rdsect'
    traces = [_ for _ in traces if "rdisosect" not in _]
    # append full random
    if not "rdisofull" in [_.split("_")[0] for _ in traces]:
        rdfull = [_ for _ in df.columns if "rdisofull" in _]
        traces.extend(rdfull)
    # filter -> only significative cells
    traces = [_ for _ in traces if not _.endswith("sig")]
    # text labels
    title = "{} {}".format(titles.get(rec, ""), titles.get(spread, ""))
    anotx = "Cell Rank"
    if age == "old":
        anoty = [r"$\Delta$ Phase (ms)", r"$\Delta$ Amplitude"]
        # (fraction of Center-only response)']
    else:
        anoty = [titles["time50"], titles.get(amp, "")]

    # plot
    fig, axes = plt.subplots(
        4, 2, figsize=(12.5, 18), sharex=True, sharey="col", squeeze=False
    )  # •sharey=True,
    if anot:
        fig.suptitle(title, alpha=0.4)
    axes = axes.flatten()
    x = range(1, len(df) + 1)
    # use cpisotime for ref
    trace = traces[key]
    sig_trace = trace + "_sig"
    df = df.sort_values(by=[trace, sig_trace], ascending=False)
    # plot all traces
    for i, trace in enumerate(traces):
        sig_trace = trace + "_sig"
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0: (1, 1, 1), 1: edge_color}
        if sort_all:
            select = df[[trace, sig_trace]].sort_values(
                by=[trace, sig_trace], ascending=False
            )
        else:
            select = df[[trace, sig_trace]]
        bar_colors = [color_dic[x] for x in select[sig_trace]]
        ax = axes[i]
        # ax.set_title(str(i))
        ax.bar(
            x,
            select[trace],
            color=bar_colors,
            edgecolor=edge_color,
            alpha=0.8,
            width=0.8,
        )
        # test to avoid to much bars width
        # if i % 2 == 0:
        #     ax.bar(x, select[name], color=bar_colors, edgecolor=edge_color,
        #            alpha=0.8, width=0.8)
        # else:
        #     ax.bar(x, select[name], color=edge_color, edgecolor=edge_color,
        #            alpha=0.8, width=0.1)
        #     ax.scatter(x, select[name].values, c=bar_colors, marker='o',
        #                edgecolor=edge_color, s=82)
        if i in [0, 1]:
            ax.set_title(anoty[i])
    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[::2]
    right_axes = axes[1::2]
    for axe in [left_axes, right_axes]:
        for i, ax in enumerate(axe):
            ax.set_facecolor((1, 1, 1, 0))
            # ax.set_title(i)
            ax.spines["top"].set_visible(False)
            # ax.ticklabel_format(useOffset=True)
            ax.spines["bottom"].set_visible(False)
            # zero line
            ax.axhline(0, alpha=0.3, color="k")
            if i != 3:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel(anotx)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xticks([1, len(df)])
                ax.set_xticklabels([1, len(df)])
                ax.set_xlim(0, len(df) + 1)
    for ax in left_axes:
        custom_ticks = np.linspace(0, 10, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    for ax in right_axes:
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
    no_spines = True
    if no_spines is True:
        for ax in left_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 10, color="k", linewidth=2)
            for spine in ["left", "right"]:
                ax.spines[spine].set_visible(False)
        for ax in right_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color="k", linewidth=2)
            # ax.axvline(limx[1], 0, -0.5, color='k', linewidth=2)
            for spine in ["left", "right"]:
                ax.spines[spine].set_visible(False)

    # align each row yaxis on zero between subplots
    gfunc.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gfunc.change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots
    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.5, wspace=0.2)
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "sorted.py:plot_all_cg_sorted_responses",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        fig.text(0.5, 0.01, "sorted", ha="left", va="bottom", alpha=0.4)
    return fig


indexed_df = load_measures()
fig = plot_all_cg_sorted_responses(indexed_df)

#%%
