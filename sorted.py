#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot centrigabor sorted responses
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

#%% save data for fig7a
def save7a(do_save=False):
    """save the data for the figure 7"""
    df = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")
    cols = df.columns

    data_savename = os.path.join(paths["figdata"], "fig7.hdf")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "sort"))
    for item in cols:
        print(item)
    print()
    if do_save:
        df.to_hdf(data_savename, key="sort")


save7a(False)

#%% plot latency (left) and gain (right)


# def plot_all_cg_sorted_responses(
#     overlap=True, sort_all=True, key=0, spread="sect", rec="vm", age="new", amp="engy"
# ):
def plot_all_cg_sorted_responses(**kwargs):
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
    df = ldat.load_cell_contributions(rec=rec, amp=amp, age=age)
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


# old
# fig = plot_sorted_responses_sup1(overlap=True)
# fig = plot_sorted_responses_sup1(overlap=True, sort_all=False)
#%%
plt.close("all")

record = ["vm", "spk"][0]

# figure = plot_all_sorted_responses(overlap=True, sort_all=False,
#                                  rec=record, amp='engy', age='new')

figure = plot_all_cg_sorted_responses(
    overlap=True, sort_all=True, rec=record, amp="engy", age="new"
)
save = False
if save:
    name = "f7_all_cg_sorted_responses"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["save"], (name + ext)))

# figure = plot_all_sorted_responses(overlap=True, sort_all=False, key=1,
#                                  rec=record, amp='engy', age='new')
#%%
plt.close("all")
save = False
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "sorted", "sorted&contrib"
)
amplitude = "engy"
for record in ["vm", "spk"]:
    for spread_space in ["sect", "full"]:
        # for amp in ['gain', 'engy']:
        figure = plot_all_cg_sorted_responses(
            overlap=True,
            sort_all=True,
            rec=record,
            amp=amplitude,
            age="new",
            spread=spread_space,
        )
        if save:
            file = record + spread_space.title() + "_" + amplitude + ".pdf"
            filename = os.path.join(paths["save"], file)
            figure.savefig(filename, format="pdf")
            # current implementation
            if record == "vm" and spread_space == "sect":
                folder = os.path.join(
                    paths["owncFig"], "pythonPreview", "current", "fig"
                )
                filename = os.path.join(folder, file)
                figure.savefig(filename, format="pdf")


# =============================================================================
# savePath = os.path.join(paths['cgFig'], 'pythonPreview', 'sorted', 'testAllSortingKeys')
# for key in range(10):
#     figure = plot_sorted_responses_sup1(overlap=True, sort_all=False, key=key)
#     filename = os.path.join(savePath, str(key) + '.png')
#     figure.savefig(filename, format='png')

# =============================================================================

#%% test boxplot
# df = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")

# conds = list(dict.fromkeys([_.split("_")[0] for _ in df.columns if mes in _]))

plt.close("all")


def load_pop():
    """load the population data for statistical display
    input:
        empty
    output:
        popdf : pandasDataframe with time and engy values
        sigdf : dictionary of the 3sig_cells for each condition
    """
    popdf = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")
    # significative cells
    conds = list(dict.fromkeys([_.split("_")[0] for _ in popdf.columns]))
    # nb sig cells <-> sig for any of conditions
    sigcells = {}
    for cond in conds:
        sigcells[cond] = popdf[
            popdf[[_ for _ in popdf.columns if cond in _ if "_sig" in _]].sum(axis=1)
            > 0
        ].index.to_list()
    cols = popdf.columns
    # pop
    cols = [_ for _ in popdf.columns if "_sig" not in _]
    popdf = popdf[cols]
    return popdf, sigcells


def boxPlot(popdf, sigcells, sigonly=False):
    """draw a boxplot:
    input:
        popdf : pandasDataframe
        sigcells : dictionary of the significant cells (list) per condition (key)
        sigonly : boolean True <-> sigPopulation, False <-> all cells
    output:
        matplotlib.pyplot figure
    """
    titles = config.std_titles()
    df = popdf.copy()
    # pop <-> all the cells : update the sigcell dictionary
    if sigonly:
        selected_cells = sigcells.copy()
    else:
        selected_cells = {k: popdf.index.to_list() for k, v in sigcells.items()}
    # build canvas
    fig = plt.figure()
    axes = []
    ax0 = fig.add_subplot(221)
    axes.append(ax0)
    axes.append(fig.add_subplot(223, sharey=ax0))
    ax1 = fig.add_subplot(222)
    axes.append(ax1)
    axes.append(fig.add_subplot(224, sharey=ax1))
    # select data
    colors = [std_colors[_] for _ in ["red", "green", "yellow", "blue"]]
    measures = ["time50", "engy"]
    spreads = ["sect", "full"]
    keys = [(a, b) for a in measures for b in spreads]
    # iterate on the plots
    for i, (mes, spread) in enumerate(keys):
        cols = [_ for _ in df.columns if mes in _ if spread in _ if "_sig" not in _]
        # extract data
        to_plot = pd.DataFrame()
        for col in cols:
            cells = selected_cells[col.split("_")[0]]
            ser = pd.Series(name=col, data=df.loc[cells][col].values)
            to_plot = pd.concat([to_plot, ser], ignore_index=True, axis=1)
        to_plot.columns = cols
        # drop nan
        data = to_plot.values
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
        # plot
        ax = axes[i]
        if i < 2:
            bp = ax.boxplot(
                filtered_data,
                notch=True,
                vert=False,
                patch_artist=True,
                showcaps=False,
                widths=0.7,
            )
        else:
            bp = ax.boxplot(
                filtered_data,
                notch=True,
                vert=True,
                patch_artist=True,
                showcaps=False,
            )

        for j, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=colors[j], alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_title(titles[measures[0]])
    axes[0].set_ylabel(titles[spreads[0]])
    axes[1].set_ylabel(titles[spreads[1]])
    axes[2].set_title(titles[measures[1]])

    conds = [_.split("_")[0][:-4] for _ in cols]
    axes[0].set_yticklabels(conds * 2)
    axes[-1].set_xticklabels(conds)
    axes[0].set_xlim(axes[0].get_xlim()[0], 20)
    axes[1].set_xlim(axes[0].get_xlim()[0], 20)

    axes[0].set_xticklabels([])
    axes[2].set_xticklabels([])
    for ax in axes[:2]:
        ax.axvline(0, alpha=0.3, color="k")
    for ax in axes[2:]:
        ax.axhline(0, alpha=0.3, color="k")

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "sorted.py:boxPlot",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        if sigonly:
            fig.text(0.5, 0.01, "sigPop", ha="left", va="bottom", alpha=0.4)
        else:
            fig.text(0.5, 0.01, "pop", ha="left", va="bottom", alpha=0.4)

    fig.tight_layout()
    return fig


if not "pop_df" in dir():
    pop_df, sig_cells = load_pop()


save = False
for sig_only in [True, False]:
    figure = boxPlot(pop_df, sig_cells, sigonly=sig_only)
    if save:
        if sig_only:
            file = "boxplot_sigPop.pdf"
        else:
            file = "boxplot_pop.pdf"
        dirname = os.path.join(paths["owncFig"], "pythonPreview", "current", "figSup")
        figure.savefig(os.path.join(dirname, file))

#%%
plt.close("all")


def violin_plot(popdf, sigcells, sigonly=False):
    """draw a violin plot:
    input:
        popdf : pandasDataframe
        sigcells : dictionary of the significant cells (list) per condition (key)
        pop : boolean True <-> allPopulation, False <-> only significant cells
    output:
        matplotlib.pyplot figure
    """
    titles = config.std_titles()
    df = popdf.copy()

    # update the (sig)cell dictionary
    if sigonly:
        selected_cells = sigcells.copy()
    else:
        selected_cells = {k: popdf.index.to_list() for k, v in sigcells.items()}
    # build canvas
    fig = plt.figure()
    axes = []
    ax0 = fig.add_subplot(221)
    axes.append(ax0)
    axes.append(fig.add_subplot(223, sharey=ax0))
    ax1 = fig.add_subplot(222)
    axes.append(ax1)
    axes.append(fig.add_subplot(224, sharey=ax1))
    # select data
    colors = [std_colors[_] for _ in ["red", "green", "yellow", "blue"]]
    measures = ["time50", "engy"]
    spreads = ["sect", "full"]
    # iterate on the plots
    keys = [(a, b) for a in measures for b in spreads]
    for i, (mes, spread) in enumerate(keys):
        cols = [_ for _ in df.columns if mes in _ if spread in _ if "_sig" not in _]
        # build dataframe with values per condition
        to_plot = pd.DataFrame()
        for col in cols:
            cells = selected_cells[col.split("_")[0]]
            ser = pd.Series(name=col, data=df.loc[cells][col].values)
            to_plot = pd.concat([to_plot, ser], ignore_index=True, axis=1)
        to_plot.columns = cols
        # drop nan
        data = to_plot.values
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
        ax = axes[i]
        if i < 2:
            violin = ax.violinplot(
                filtered_data,
                vert=False,
                widths=0.7,
                showmedians=True,
            )
        else:
            violin = ax.violinplot(
                filtered_data,
                vert=True,
                widths=0.5,
                showmedians=True,
            )
        for bd, color in zip(violin["bodies"], colors):
            bd.set_facecolor(color)
            bd.set_edgecolor("k")
            bd.alpha = 0.5
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_title(titles[measures[0]])
    axes[0].set_ylabel(titles[spreads[0]])
    axes[1].set_ylabel(titles[spreads[1]])
    axes[2].set_title(titles[measures[1]])

    conds = [_.split("_")[0][:-4] for _ in cols]
    axes[0].set_yticks(range(1, 1 + len(conds)))
    axes[0].set_yticklabels(conds)
    axes[-1].set_xticks(range(1, 1 + len(conds)))
    axes[-1].set_xticklabels(conds)
    axes[0].set_xlim(axes[0].get_xlim()[0], 20)
    axes[1].set_xlim(axes[0].get_xlim()[0], 20)

    axes[0].set_xticklabels([])
    axes[2].set_xticklabels([])
    for ax in axes[:2]:
        ax.axvline(0, alpha=0.3, color="k")
    for ax in axes[2:]:
        ax.axhline(0, alpha=0.3, color="k")

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "sorted.py:violin_plot",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        if sigonly:
            fig.text(0.5, 0.01, "sigPop", ha="left", va="bottom", alpha=0.4)
        else:
            fig.text(0.5, 0.01, "pop", ha="left", va="bottom", alpha=0.4)

    fig.tight_layout()
    return fig


if not "pop_df" in dir():
    pop_df, sig_cells = load_pop()


save = False
for sig_only in [True, False]:
    figure = violin_plot(pop_df, sig_cells, sigonly=sig_only)
    if save:
        if sig_only:
            file = "violinplot_sigPop.pdf"
        else:
            file = "violinplot_pop.pdf"
        dirname = os.path.join(paths["owncFig"], "pythonPreview", "current", "figSup")
        figure.savefig(os.path.join(dirname, file))
