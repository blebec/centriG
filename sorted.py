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


def plot_all_cg_sorted_responses(
    overlap=True, sort_all=True, key=0, spread="sect", kind="vm", age="new", amp="engy"
):
    """
    plot the sorted cell responses
    input = conditions parameters
    overlap : boolean, overlap the different rows to superpose the plots
    sort_all : if false, only the 'key' trace is sorted
    key : number to choose the trace to be sorted
    output : matplotlib plot
    """

    def set_ticks_both(axis):
        """ set ticks and ticks labels on both sides """
        ticks = list(axis.majorTicks)  # a copy
        ticks.extend(axis.minorTicks)
        for t in ticks:
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
            t.label1.set_visible(True)
            t.label2.set_visible(True)

    titles = {
        "engy": r"$\Delta$ Response",
        "time50": r"$\Delta$ Latency",
        "gain50": "Amplitude Gain",
        "sect": "Sector",
        "spk": "Spikes",
        "vm": "Vm",
        "full": "Full",
    }

    # parameter
    colors = [std_colors[item] for item in ["red", "green", "yellow", "blue", "blue"]]
    colors = [color for color in colors for _ in range(2)]
    # data (call)
    df = ldat.load_cell_contributions(rec=kind, amp=amp, age=age)
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if spread in item]
    # remove the 'rdsect'
    traces = [item for item in traces if "rdisosect" not in item]
    # append full random
    if not "rdisofull" in [item.split("_")[0] for item in traces]:
        rdfull = [item for item in df.columns if "rdisofull" in item]
        traces.extend(rdfull)
    # filter -> only significative cells
    traces = [item for item in traces if not item.endswith("sig")]
    # text labels
    title = "{} {}".format(titles.get(kind, ""), titles.get(spread, ""))
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
    name = traces[0]
    name = traces[key]
    sig_name = name + "_sig"
    df = df.sort_values(by=[name, sig_name], ascending=False)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + "_sig"
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0: (1, 1, 1), 1: edge_color}
        if sort_all:
            select = df[[name, sig_name]].sort_values(
                by=[name, sig_name], ascending=False
            )
        else:
            select = df[[name, sig_name]]
        bar_colors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        # ax.set_title(str(i))
        ax.bar(
            x,
            select[name],
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
    if no_spines == True:
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

kind = ["vm", "spk"][0]

# fig = plot_all_sorted_responses(overlap=True, sort_all=False,
#                                  kind=kind, amp='engy', age='new')

fig = plot_all_cg_sorted_responses(
    overlap=True, sort_all=True, kind=kind, amp="engy", age="new"
)
save = False
if save:
    name = "f7_all_cg_sorted_responses"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(os.path.join(paths["save"], (name + ext)))

# fig = plot_all_sorted_responses(overlap=True, sort_all=False, key=1,
#                                  kind=kind, amp='engy', age='new')
#%%
plt.close("all")
save = False
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "sorted", "sorted&contrib"
)
amp = "engy"
for kind in ["vm", "spk"]:
    for spread in ["sect", "full"]:
        # for amp in ['gain', 'engy']:
        fig = plot_all_cg_sorted_responses(
            overlap=True, sort_all=True, kind=kind, amp=amp, age="new", spread=spread
        )
        if save:
            file = kind + spread.title() + "_" + amp + ".pdf"
            filename = os.path.join(paths["save"], file)
            fig.savefig(filename, format="pdf")
            # current implementation
            if kind == "vm" and spread == "sect":
                folder = os.path.join(
                    paths["owncFig"], "pythonPreview", "current", "fig"
                )
                filename = os.path.join(folder, file)
                fig.savefig(filename, format="pdf")


# =============================================================================
# savePath = os.path.join(paths['cgFig'], 'pythonPreview', 'sorted', 'testAllSortingKeys')
# for key in range(10):
#     fig = plot_sorted_responses_sup1(overlap=True, sort_all=False, key=key)
#     filename = os.path.join(savePath, str(key) + '.png')
#     fig.savefig(filename, format='png')

# =============================================================================

#%% test boxplot
# df = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")

# conds = list(dict.fromkeys([_.split("_")[0] for _ in df.columns if mes in _]))

plt.close("all")


def load_pop():
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


def boxPlot(popdf, sigcells, pop=False):

    titles = {
        "engy": r"$\Delta$ Response",
        "time50": r"$\Delta$ Latency",
        "gain50": "Amplitude Gain",
        "sect": "Sector",
        "spk": "Spikes",
        "vm": "Vm",
        "full": "Full",
    }
    df = popdf.copy()
    scells = sigcells.copy()
    if pop:
        for k in scells:
            scells[k] = popdf.index.to_list()

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # axes = axes.flatten('F')

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

    for i, (mes, spread) in enumerate(keys):
        cols = [_ for _ in df.columns if mes in _ if spread in _ if "_sig" not in _]
        toPlot = pd.DataFrame()
        for col in cols:
            cells = scells[col.split("_")[0]]
            ser = pd.Series(name=col, data=df.loc[cells][col].values)
            toPlot = pd.concat([toPlot, ser], ignore_index=True, axis=1)
        toPlot.columns = cols
        # drop nan
        data = toPlot.values
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d, m in zip(data.T, mask.T)]

        ax = axes[i]
        if i < 2:
            bp = ax.boxplot(
                filtered_data,
                notch=True,
                vert=False,
                # meanline=True,
                # showmeans=True,
                patch_artist=True,
                showcaps=False,
            )
        else:
            bp = ax.boxplot(
                filtered_data,
                notch=True,
                vert=True,
                # meanline=True,
                # showmeans=True,
                patch_artist=True,
                showcaps=False,
            )

        for j, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=colors[j], alpha=0.3)
        #    ax.set_xticklabels(toPlot.columns.to_list(), rotation=45, ha="right")
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
            0.99, 0.01, "sorted.py:boxPlot", ha="right", va="bottom", alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        if pop:
            fig.text(0.5, 0.01, "pop", ha="left", va="bottom", alpha=0.4)
        else:
            fig.text(0.5, 0.01, "sigPop", ha="left", va="bottom", alpha=0.4)

    fig.tight_layout()
    return fig


if not "pop_df" in dir():
    pop_df, sig_cells = load_pop()


save = False
for pop in [True, False]:
    fig = boxPlot(pop_df, sig_cells, pop=pop)
    if save:
        if pop:
            file = "boxplot_pop.pdf"
        else:
            file = "boxplot_sigPop.pdf"
        dirname = os.path.join(paths["owncFig"], "pythonPreview", "current", "figSup")
        fig.savefig(os.path.join(dirname, file))

#%% replace nonsig by nan values
