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
from matplotlib.patches import Rectangle
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
    """load from hdf file the measure ie index data"""
    key = "indexes"
    data_loadname = os.path.join(paths["figdata"], "measures.hdf")
    df = pd.read_hdf(data_loadname, key=key)
    conds = {_.split("_")[0] for _ in df.columns}
    dico = {}
    for cond in conds:
        dico[cond] = ["_".join(_.split("_")[1:]) for _ in df.columns if cond in _]
    print("=" * 20, "{}(key={})".format(os.path.basename(data_loadname), key))
    if display:
        for k, v in dico.items():
            print(k, v)
    print()
    return df


if "stat_df" not in dir():
    stat_df = ldat.build_pop_statdf()
if "stat_df_sig" not in dir():
    stat_df_sig, sig_cells = ldat.build_sigpop_statdf()
if "indexed_df" not in dir():
    indexed_df = load_measures()

#%%

plt.close("all")


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
        """set ticks and ticks labels on both sides"""
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
    size = (13, 10.6)
    fig, axes = plt.subplots(
        4, 2, figsize=(13, 10.6), sharex=True, sharey="col", squeeze=False
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


# indexed_df = load_measures()
figure = plot_all_cg_sorted_responses(indexed_df)

save = False
if save:
    file = "f7_sorted"
    # paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".pdf"]:  # [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["figSup"], (file + ext)))
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        figure.savefig(filename)

#%%
def plot_composite_stat(
    statdf,
    statdfsig,
    sigcells,
):
    """
    plot the stats
    input : statdf, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    here combined top = full pop, bottom : significative pop
    """

    # to build
    statdf = stat_df
    statdfsig = stat_df_sig
    sigcells = sig_cells

    kind = "mean"
    amp = "engy"
    mes = "vm"
    legend = False
    share = True
    digit = True

    kinds = {"mean": ["_mean", "_sem"], "med": ["_med", "_mad"]}
    stat = kinds.get(kind, None)
    titles = config.std_titles()
    # colors = [std_colors[_] for _ in ["red", "green", "yellow", "dark_blue", "blue"]]
    colors = [std_colors[_] for _ in ["red", "green", "yellow", "blue", "dark_blue"]]

    # share scales high and low
    # fig, axes = plt.subplots(
    #     nrows=1, ncols=3, figsize=(16, 5), sharex=True, sharey=True
    # )
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True
    )
    axes = axes.flatten()
    title = stat[0][1:] + "   (" + stat[1][1:] + ")"
    fig.suptitle(title)
    # conds = [(x, y) for x in [mes, mes] for y in ["sect", "full"]]
    conds = [("vm", "sect"), ("vm", "sect"), ("vm", "full")]
    conds = conds[:2]
    for i, cond in enumerate(conds):
        if i >= 1:
            df = statdfsig
            pop = "sig_pop"
        else:
            df = statdf
            pop = "pop"
        ax = axes[i]
        rec, spread = cond
        ax_title = "{} ({} {})".format(pop, rec, spread)
        ax.set_title(ax_title)
        # select spread (sect, full)
        rows = [_ for _ in df.index.tolist() if spread in _]
        # append random full if sector
        if spread == "sect":
            # rows.extend([_ for _ in stat_df.index if _.startswith("rdisofull")])
            # insert full stim before sectstim
            fulls = [_ for _ in stat_df.index if _.startswith("rdisofull")]
            if fulls:
                sec_pos = rows.index([_ for _ in rows if "rd" in _][0])
                for j, full in enumerate(fulls):
                    rows.insert(sec_pos + j, full)
        # df indexes (for x and y)
        time_rows = [_ for _ in rows if "time50" in _]
        y_rows = [_ for _ in rows if amp in _]
        # y_rows = [st for st in rows if 'engy' in st]
        cols = [_ for _ in df.columns if _.startswith(rec)]
        cols = [_ for _ in cols if stat[0] in _ or stat[1] in _]
        # labels
        labels = [_.split("_")[0] for _ in y_rows]
        # values (for x an y)
        x = df.loc[time_rows, cols][rec + stat[0]].values
        xerr = df.loc[time_rows, cols][rec + stat[1]].values
        y = df.loc[y_rows, cols][rec + stat[0]].values
        yerr = df.loc[y_rows, cols][rec + stat[1]].values
        # plot
        if not digit:
            # marker in the middle
            for xi, yi, xe, ye, ci, lbi in zip(x, y, xerr, yerr, colors, labels):
                ax.errorbar(xi, yi, xerr=xe, yerr=ye, fmt="s", color=ci, label=lbi)
        else:
            # nb of cells in the middle
            # extract nb of cells
            key = "_".join([rec, "count"])
            cell_nb = [int(df.loc[item, [key]][0]) for item in y_rows]
            for xi, yi, xe, ye, ci, lbi, nbi in zip(
                x, y, xerr, yerr, colors, labels, cell_nb
            ):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=xe,
                    yerr=ye,
                    # fmt="s",
                    color=ci,
                    label=lbi,
                    marker="s",
                    ms=16,
                    mec="w",
                    mfc="w",
                )
                # marker='$'+ str(nbi) + '$', ms=24, mec='w', mfc=ci)
                ax.text(
                    xi,
                    yi,
                    str(nbi),
                    color=ci,
                    fontsize=14,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        if legend:
            ax.legend()
    # adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle="-", alpha=0.4, color="k")
        ax.axhline(0, linestyle="-", alpha=0.4, color="k")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            # # horizontal
            #     if i == 0:
            #         ax.set_ylabel(titles.get(amp, "not defined"))
            #     else:
            #         ax.spines["left"].set_visible(False)
            #         ax.yaxis.set_visible(False)
            #     ax.set_xlabel(titles["time50"])
            # vertical
            ax.set_xlabel(titles["time50"])
            if i == 0:
                ax.set_ylabel(titles.get(amp, "not defined"))
    #          else:
    #               ax.spines["left"].set_visible(False)
    #                ax.yaxis.set_visible(False)

    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.02)
    # fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "fig7_indexes.py:plot_composite_stat_3x1",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


plt.close("all")
# stat_df = ldat.build_pop_statdf(amp=amplitude)  # append gain to load
# stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amplitude)  # append gain to load
figure = plot_composite_stat(stat_df, stat_df_sig, sig_cells)

save = False
if save:
    file = "f7_stats"
    folder = paths["figSup"]
    # paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".pdf"]:  # [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(folder, (file + ext)))
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        figure.savefig(filename)


#%%
def extract_values(df, stim="sect", param="time", replaceFull=True):
    """extract pop and response dico:
    input :
        dataframe
        stim in [sect, full]
        param in [timme, gain, engy]
    return:
        pop_dico -> per condition [popNb, siginNb, %]
        resp_dico -> per condition [moy, moy+sem, moy-sem]
    """
    adf = df.copy()
    if "fill" in param:
        fills = [item for item in adf.columns if "fill" in item]
        # create zero padded value columns
        while fills:
            fill = fills.pop()
            col = "_".join(fill.split("_")[:-1])
            if col not in adf.columns:
                adf[col] = data[fill]
                adf[col] = 0
    restricted_list = [st for st in adf.columns if stim in st and param in st]
    if replaceFull:
        # replace rdisosect by rdisofull
        if "rdisosect" in set(item.split("_")[0] for item in restricted_list):
            restricted_list = [
                st for st in restricted_list if "rdisosect" not in st.split("_")[0]
            ]
            restricted_list.extend(
                [st for st in adf.columns if "rdisofull" in st and param in st]
            )
    else:
        # append full:
        if "rdisosect" in {item.split("_")[0] for item in restricted_list}:
            restricted_list.extend(
                [st for st in adf.columns if "rdisofull" in st and param in st]
            )

    adf = adf[restricted_list]
    # compute values
    # records = [item for item in restricted_list if 'sig' not in item]
    # to maintain the order
    records = [item.replace("_sig", "") for item in restricted_list if "sig" in item]

    pop_dico = {}
    resp_dico = {}
    for cond in records:
        signi = cond + "_sig"
        pop_num = len(adf)
        # significant
        extract = adf.loc[adf[signi] > 0, cond].copy()
        # on ly positive measures
        extract = extract[extract >= 0]
        signi_num = len(extract)
        percent = round((signi_num / pop_num) * 100)
        leg_cond = cond.split("_")[0]
        pop_dico[leg_cond] = [pop_num, signi_num, percent]
        # descr
        moy = extract.mean()
        sem = extract.sem()
        resp_dico[leg_cond] = [moy, moy + sem, moy - sem]
    return pop_dico, resp_dico


def autolabel(ax, rects, sup=False):
    """
    attach the text labels to the rectangles
    """
    for rect in rects:
        x = rect.get_x() + rect.get_width() / 2
        height = rect.get_height()
        y = height - 1
        if y < 3 or sup:
            y = height + 1
            ax.text(x, y, "%d" % int(height) + "%", ha="center", va="bottom")
        else:
            ax.text(x, y, "%d" % int(height) + "%", ha="center", va="top")


def plot_cell_selection(df, sigcells, spread="sect", mes="vm", amp="engy"):
    """
    cell contribution, to go to the bottom of the preceding stat description
    """

    # to build
    df = data
    sigcells = sig_cells
    spread = "sect"
    mes = "vm"
    amp = "engy"

    titles = dict(
        time=r"$\Delta$ Latency",
        engy=r"$\Delta$ Response",
        sect="Sector",
        full="Full",
        vm="Vm",
        spk="Spikes",
    )
    colors = [std_colors[item] for item in "red green yellow blue dark_blue".split()]
    relabel = dict(
        cpisosect="CP-ISO",
        cfisosect="CF-ISO",
        cpcxsect="CP-CROSS",
        rdisosect="RND",
        cpisofull="CP-ISO",
        cfisofull="CF-ISO",
        cpcxfull="CP-CROSS",
        rdisofull="RND",
    )
    # compute values ([time values], [amp values])
    params = ["time", amp]
    heights = []
    for param in params:
        pop_dico, _ = extract_values(df, spread, param)
        height = [pop_dico[key][-1] for key in pop_dico]
        heights.append(height)
    # insert union % sig cells for time and amp
    height = [
        round(len(sigcells[mes][st]) / len(df) * 100) for st in list(pop_dico.keys())
    ]
    heights.insert(1, height)

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18, 3.75))
    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(5, 12.5))
    axes = axes.flatten()
    titles_here = [titles["time"], "Both", titles["engy"]]
    labels = [relabel[st] for st in pop_dico]

    for i, ax in enumerate(axes):
        ax.axhline(0, color="k")
        for spine in ["left", "top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
            # ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(str(i))
        param = params[0]
        # for param in params
        ax.set_title(titles_here[i], pad=0)

        x = np.arange(len(heights[i]))
        width = 0.95
        if i in [0, 2]:
            bars = ax.bar(
                x, heights[i], color=colors, width=width, alpha=0.6, edgecolor="k"
            )
            # edgecolor=colors)
        else:
            bars = ax.bar(
                x,
                heights[i],
                color=colors,
                width=width,
                alpha=0.9,
                edgecolor="k",
                label=labels,
            )
        autolabel(ax, bars)  # call
        # labels = list(pop_dico.keys())
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in axes:
        ax.set_ylabel(r"% of significant cells")
    # for ax in axes[:2]:
    #     ax.xaxis.set_visible(False)
    # fig.legend(handles=bars, labels=labels, loc='upper right')
    # rectangle
    box = True
    if box:
        ax = axes[1]
        x, x1 = ax.get_xlim()
        step = (x1 - x) * 0.4
        x1 -= x
        y, y1 = ax.get_ylim()
        y1 -= y
        y = 0 - step
        rect = Rectangle(
            xy=(x, y),
            width=x1,
            height=y1 + step,
            fill=False,
            alpha=0.6,
            edgecolor="k",
            linewidth=6,
        )
        ax.add_patch(rect)
        ax.set_ylim(y, y1)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            1,
            0.01,
            "cellContribution:plot_cell_selection",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        txt = "select      {} {} ({} cells) ".format(mes, spread, len(data))
        fig.text(0.5, 0.01, txt, ha="center", va="bottom", fontsize=14, alpha=0.4)

    fig.tight_layout(w_pad=4)
    return fig


plt.close("all")
save = False
amp = ["gain", "engy"][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)
mes = ["vm", "spk"][0]
data = ldat.load_cell_contributions(mes, age="new", amp=amp)
spread = ["sect", "full"][0]
figure = plot_cell_selection(data, sig_cells, spread=spread, mes=mes, amp=amp)

save = False
if save:
    file = "f7_contrib"
    # paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".pdf"]:  # [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["figSup"], (file + ext)))
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        figure.savefig(filename)
