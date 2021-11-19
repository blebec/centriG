#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot centrigabor figures from data stored in .xlsx files
"""

import os
from datetime import datetime
from importlib import reload

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pandas.plotting import table

import config
import fig_proposal as figp
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra
import old.old_figs as ofig

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
os.chdir(paths["pg"])

# energy_df = ldat.load_energy_gain_index(paths)
# latGain50_v_df = ldat.load_cell_contributions(rec='vm')
# latGain50_s_df = ldat.load_cell_contributions(rec='spk')


def get_sig3_df(key="sector"):
    """ get sig amp U time U fill data """

    if key != "sector":
        print("{} should be implemented".format(key))
        return
    filename = os.path.join(
        paths["owncFig"],
        "data/averageTraces/controlsFig/union_idx_fill_sig_sector.xlsx",
    )
    sig3df = pd.read_excel(filename, engine="openpyxl")
    cols = gfunc.new_columns_names(sig3df.columns)
    cols = [item.replace("sig_", "") for item in cols]
    cols = [item.replace("_stc", "") for item in cols]
    cols = [st.replace("_iso", "") for st in cols]
    cols = [st.replace("__", "_") for st in cols]
    cols = [st.replace("_.1", "") for st in cols]
    sig3df.columns = cols
    # adjust time scale
    middle = (sig3df.index.max() - sig3df.index.min()) / 2
    sig3df.index = (sig3df.index - middle) / 10
    return sig3df


sig3_df = get_sig3_df()


#%%
plt.close("all")


def plot_cpIsoGain(
    datadf, sig3df, colsdict, anot=False, age="new", stdcolors=std_colors
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
    # # remove sig 'only pop value (old sig = whithout fill+)
    # cols = [_ for _ in datadf.columns if _.endswith("Sig")]
    # # sigdf = datadf[cols].copy()
    # datadf.drop(labels=cols, axis=1, inplace=True)
    # # left column = example
    # cols = [_ for _ in datadf.columns if _.startswith("indi")]
    # indidf = datadf[cols].copy()
    # # middle column = pop
    # cols = [_ for _ in datadf.columns if _.startswith("pop")]
    # popdf = datadf[cols].copy()
    popdf = datadf.copy()
    # data for the figure right column = sigDf
    # NB (sigpop= sigTime u sigAmpl u sigFill)
    fill = True
    colors = [stdcolors[_] for _ in "k red".split()]
    alphas = (0.8, 0.8)
    vspread = 0.06  # vertical spread for realign location

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

    # ___ individual vm
    # cols = colsdict["indVm"]
    cols = popdf.columns
    cols = [_ for _ in cols if _.startswith("indi") and "Vm" in _]
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(popdf[col], color=colors[i], alpha=alphas[i], label=col)
    # response point
    x_pos = dict(old=41.5, new=43.5)
    x = x_pos[age]
    y = datadf.indiVmctr.loc[x]
    # blue point and vline
    #    ax.plot(x, y, 'o', color=std_colors['blue'], ms=10, alpha=0.8)
    ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ individual spike
    cols = popdf.columns
    cols = [_ for _ in cols if _.startswith("indi") and "Spk" in _]
    ax = spkaxes[0]
    rev_cols = cols[::-1]
    rev_colors = colors[::-1]
    df = popdf[rev_cols].copy()
    for i, col in enumerate(rev_cols):
        ax.plot(df[col], color=rev_colors[i], alpha=1, label=col, linewidth=1)
        ax.fill_between(df.index, df[col], color=rev_colors[i], alpha=0.5, label=col)
    # response point
    x_pos = dict(old=39.8, new=55.5)
    x = x_pos[age]
    y = datadf.indiSpkCtr.loc[x]
    ax.plot(x, y, "o", color="tab:blue", ms=10, alpha=0.8)
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ pop vm
    df = popdf.loc[-20:60]  # limit xscale
    cols = df.columns
    cols = [_ for _ in cols if _.startswith("pop") and "Vm" in _]
    cols = [_ for _ in cols if "Sig" not in _]
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=1.5)
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
    y = df[cols[0]].loc[x]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # ___ pop spike
    cols = df.columns
    cols = [_ for _ in cols if _.startswith("pop") and "Spk" in _]
    cols = [_ for _ in cols if "Sig" not in _]
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=rev_colors[i], alpha=1, label=col, linewidth=1.5)
        # ax.fill_between(popdf.index, df[col],
        #                 color=colors[::-1][i], alpha=0.5, label=col)
    ax.annotate(
        "n=22",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="center",
        va="top",
    )
    # response point
    x = 0
    y = df[cols[0]].loc[x]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # popVmSig
    cols = colsdict["popVmSig"]
    ax = vmaxes[2]
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col)
        # errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(
                    df.index, df[col[0]], df[col[1]], color=colors[i], alpha=0.2
                )  # alphas[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(
                            df[col[j]],
                            color=colors[i],
                            alpha=alphas[i],
                            label=col,
                            linewidth=0.5,
                        )
    # response point
    x = 0
    y = df[cols[0]].loc[x]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")
    ax.annotate(
        "n=10",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    # popSpkSig
    cols = colsdict["popSpkSig"]
    ax = spkaxes[2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(
            df[col],
            color=colors[::-1][i],
            alpha=alphas[::-1][i],
            label=col,
            linewidth=2,
        )
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(
            df.index, df[col[0]], df[col[1]], color=colors[i], alpha=alphas[::-1][i] / 2
        )  # label=col, linewidth=0.5)
        # for j in [0, 1]:
        #     ax.plot(df[col[j]], color=colors[i],
        #             alpha=1, label=col, linewidth=0.5)
    # response point
    x = 0
    y = df[cols[0]].loc[x]
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")
    ax.annotate(
        "n=5",
        xy=(0.9, 0.9),
        size="large",
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    ######## SIG #######
    colors = [stdcolors[color] for color in "k red green yellow blue blue".split()]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    nbcells = dict(sect=[20, 10], full=[15, 7])  # [vm, spk]

    axes = [vmaxes[-1], spkaxes[-1]]
    records = ["vm"] + ["spk"]
    # leg_dic = dict(vm="Vm", spk="Spikes")
    cols = [_ for _ in sig3df.columns if "sect" in _]
    # plot
    for i, ax in enumerate(axes):
        # rec = recs[i]
        # ax_title = f"{rec} {spread}"
        # ax.set_title(ax_title)
        record = records[i]
        # txt = leg_dic[record]
        # ax.text(
        #     0.7, 0.9, txt, ha="left", va="center", transform=ax.transAxes, size="large"
        # )
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
        traces = [_.replace("sect_rd", "full_rd") for _ in traces]

        df = sig3df.loc[-20:60][traces].copy()
        substract = False
        if substract:
            # subtract the centerOnly response (ref = df['CENTER-ONLY'])
            ref = df[df.columns[0]]
            df = df.subtract(ref, axis=0)
        # build labels
        labels = traces[:]
        labels = [item.split("_")[0] + "_" + item.split("_")[-1] for item in labels]
        labels = [item.replace("rd", "frd") for item in labels]
        # plot
        for i, col in enumerate(traces):
            ax.plot(
                df[col],
                color=colors[i],
                alpha=alphas[i],
                label=labels[i],
                linewidth=1.5,
            )
        # bluePoint
        x = 0
        y = df.loc[0][df.columns[0]]
        vspread = 0.06  # vertical spread for realign location
        # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
        ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
        ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")

    # =============================================================================
    #     # popVmSig
    #     cols = colsdict["popVmSig"]
    #     ax = vmaxes[2]
    #     # traces
    #     for i, col in enumerate(cols[:2]):
    #         ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=1.5)
    #         # errors : iterate on tuples
    #         for i, col in enumerate(cols[2:]):
    #             ax.fill_between(
    #                 df.index, df[col[0]], df[col[1]], color=colors[i], alpha=0.2
    #             )  # alphas[i]/2)
    #     # advance
    #     x0 = 0
    #     y = df.loc[x0][cols[0]]
    #     adf = df.loc[-20:0, [cols[1]]]
    #     i1 = (adf - y).abs().values.flatten().argsort()[0]
    #     x1 = adf.index[i1]
    #     # ax.plot(x0, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    #     # ax.plot(x1, y, marker=markers.CARETLEFT, color='tab:gray',
    #     #         ms=10, alpha=0.8)
    #     ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    #     ax.axvline(y, color="tab:blue", linestyle=":", linewidth=2)
    #     # ax.hlines(y, x0, x1, color=std_colors['blue'], linestyle=':', linewidth=2)
    #     ax.annotate(
    #         "n=15", xy=(0.2, 0.8), size="large", xycoords="axes fraction", ha="center"
    #     )
    #     # adv = str(x0 - x1)
    #     # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
    #     # xycoords="axes fraction", ha='center')
    # =============================================================================

    # =============================================================================
    #     # popSpkSig
    #     cols = colsdict["popSpkSig"]
    #     ax = spkaxes[2]
    #     # traces
    #     for i, col in enumerate(cols[:2][::-1]):
    #         ax.plot(df[col], color=rev_colors[i], alpha=1, label=col, linewidth=1.5)
    #     # errors : iterate on tuples
    #     for i, col in enumerate(cols[2:]):
    #         ax.fill_between(
    #             df.index, df[col[0]], df[col[1]], color=colors[i], alpha=alphas[::-1][i] / 2
    #         )  # label=col, linewidth=0.5)
    #     # advance
    #     x0 = 0
    #     y = df.loc[x0][cols[0]]
    #     adf = df.loc[-20:0, [cols[1]]]
    #     i1 = (adf - y).abs().values.flatten().argsort()[0]
    #     x1 = adf.index[i1]
    #     ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    #     # ax.plot(x0, y, 'o', color='tab:gray', ms=10, alpha=0.8)
    #     # ax.plot(x0, y, 'o', color=std_colors['blue'])
    #     # ax.plot(x1, y, marker=markers.CARETLEFT, color=std_colors['blue'],
    #     #         ms=10, alpha=0.8)
    #     ax.axvline(y, color="tab:blue", linestyle=":", linewidth=2)
    #     ax.annotate(
    #         "n=6", xy=(0.2, 0.8), size="large", xycoords="axes fraction", ha="center"
    #     )
    #     # #advance
    # =============================================================================
    # adv = str(x0 - x1)
    # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
    # xycoords="axes fraction", ha='center')
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
    if age == "old":
        custom_ticks = np.linspace(0, 15, 4, dtype=int)
    else:
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
            0.99,
            0.01,
            "centrifigs.py:plot_cpIsoGain",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        # fig.text(0.5, 0.01, 'fig2', ha='center', va='bottom', alpha=0.4)
    return fig


# =============================================================================
# NB two possible examples:
# cellule A= 1509E_CXG4, cellule B = 1427A_CXG4
# - actual
# cellA : xlim= (-200, 150)
# blue point
# Vm ctr  x = 43.5
# Spk ctr x = 55.5
#
# Vm ylim = 12
# Spk ylim = 35
#
# - other one
# cellB : xlim= -(230, 190)
#
# blue point<br />
# Vm ctr  x = 66.4
# Spk ctr x = 50.6
#
# Vm ylim = 15
# Spkylim = 110
#
# =============================================================================

plt.close("all")
# data
age = ["old", "new"][1]
if "fig2_df" not in globals():
    fig2_df, fig2_cols = ldat.load2(age)

if "sig3_df" not in dir():
    sig3_df = get_sig3_df()

fig = plot_cpIsoGain(
    datadf=fig2_df.copy(), sig3df=sig3_df, colsdict=fig2_cols, anot=anot, age=age
)
save = False
if save:
    name = "f8_cpIsoGain_alt"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(os.path.join(paths["save"], (name + ext)))

# # other views
# # plot all
# fig = ofig.plot_2_indMoySigNsig(fig2_df, fig2_cols, std_colors, anot=anot)
# # plot ind + pop
# fig = ofig.plot_2_indMoy(fig2_df, fig2_cols, std_colors, anot)
# # sig Nsig
# fig = ofig.plot_2_sigNsig(fig2_df, fig2_cols, std_colors, anot=anot)

#%% NB the length of the sorted data are not the same compared to the other traces
# filename = 'fig2cells.xlsx'
# df = pd.read_excel(filename)

plt.close("all")


def plot_figure2B(stdcolors=std_colors, sig=True, anot=anot, age="new"):
    """
    plot_figure2B : sorted phase advance and delta response
    input:
        stdcolors =  dictionary of colors
        sig = boolan : true <-> fill significatif cell responses
        anot = boolean : show anotations
        age in [old, new] : decide of the populatin to load
    """
    if age == "old":
        filename = "data/old/fig2cells.xlsx"
        print("old file fig2cells.xlsx")
        df = pd.read_excel(filename)
        rename_dict = {
            "popVmscpIsolatg": "cpisosect_time50",
            "lagIndiSig": "cpisosect_time50_sig",
            "popVmscpIsoAmpg": "cpisosect_gain50",
            "ampIndiSig": "cpisosect_gain50_sig",
        }
        df.rename(columns=rename_dict, inplace=True)
    elif age == "new":
        amp = "engy"
        latAmp_v_df = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")
        cols = latAmp_v_df.columns
        df = latAmp_v_df[[item for item in cols if "cpisosect" in item]].copy()
        df.sort_values(by=df.columns[0], ascending=False, inplace=True)
    else:
        print("fig2cells.xlsx to be updated")
        return
    vals = [item for item in df.columns if not item.endswith("_sig")]
    signs = [item for item in df.columns if item.endswith("_sig")]

    #    df.index += 1 # cells = 1 to 37
    color_dic = {0: (1, 1, 1), 1: stdcolors["red"]}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    for i, ax in enumerate(axes):
        #        colors = [color_dic[x] for x in df[signs[i]]]
        toplot = df.sort_values(by=vals[i], ascending=False)
        colors = [color_dic[x] for x in toplot[signs[i]]]
        if sig:
            axes[i].bar(
                toplot.index,
                toplot[vals[i]],
                edgecolor=stdcolors["red"],
                color=colors,
                label=vals[i],
                alpha=0.8,
                width=0.8,
            )
        else:
            axes[i].bar(
                toplot.index,
                toplot[vals[i]],
                edgecolor=stdcolors["red"],
                color=stdcolors["red"],
                label=vals[i],
                alpha=0.8,
                width=0.8,
            )
        # zero line
        ax.axhline(0, alpha=0.4, color="k")
        # ticks
        ax.set_xlim(-1, len(df))
        ax.set_xticks([0, len(df) - 1])
        ax.set_xticklabels([1, len(df)])
        ax.set_xlabel("Cell rank")
        ax.xaxis.set_label_coords(0.5, -0.025)
        if i == 0:
            txt = "Latency Advance (ms)"
            if amp == "gain":
                ylims = (-6, 29)
            else:
                ylims = (-10, 29)
                ax.set_ylim(ylims)
            ax.vlines(-1, 0, 20, linewidth=2, color="k")
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
        else:
            txt = "Latency Advance (ms)" if amp == "gain" else r"$\Delta$ Energy"
            ylims = ax.get_ylim()
            ax.vlines(-1, 0, 0.6, linewidth=2, color="k")
            custom_yticks = np.linspace(0, 0.6, 4)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(txt)
        ax.set_ylim(ylims)
        for spine in ["left", "top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
    # align zero between plots
    gfunc.align_yaxis(axes[0], 0, axes[1], 0)
    gfunc.change_plot_trace_amplitude(axes[1], 0.75)
    fig.tight_layout()
    # anot
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_figure2B",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


fig1 = plot_figure2B(std_colors, anot=anot, age="new")
fig2 = figp.plot_2B_bis(std_colors, anot=anot, age="new")

#%% save data


def save_fig8_data(fig2df, sig3df, do_save=False):
    """ export the data to an hdf file"""
    conds, key_dico = config.std_names()

    cols = fig2df.columns
    cols = ["_" + _ + "_" for _ in cols]
    cols = [_.replace("_indi", "_indi_") for _ in cols]
    cols = [_.replace("_pop", "_pop_") for _ in cols]
    cols = [_.replace("_Vm", "_vm_") for _ in cols]
    cols = [_.replace("_Spk", "_spk_") for _ in cols]
    cols = [_.replace("_Ctr", "_Ctr_") for _ in cols]
    cols = [_.replace("_ctr", "_Ctr_") for _ in cols]
    cols = [_.replace("_scp", "_s_cp_") for _ in cols]
    cols = [_.replace("_Iso", "_iso_") for _ in cols]
    cols = [_.replace("_Stc", "_Stc_") for _ in cols]
    cols = [_.replace("_SeUp", "_SeUp_") for _ in cols]
    cols = [_.replace("_SeDw", "_SeDw_") for _ in cols]
    for k, v in key_dico.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = [_.strip("_") for _ in cols]
    datadf = fig2df.copy()
    datadf.columns = cols

    # do_save = False
    data_savename = os.path.join(paths["figdata"], "fig8.hdf")

    # individual example:
    cols = datadf.columns
    cols = [_ for _ in cols if _.startswith("indi")]
    if do_save:
        datadf[cols].to_hdf(data_savename, "indi")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "indi"))
    for item in cols:
        print(item)
    print()

    # pop
    cols = datadf.columns
    cols = [_ for _ in cols if not _.startswith("indi")]
    cols = [_ for _ in cols if not "Sig" in _]
    if do_save:
        datadf[cols].to_hdf(data_savename, "pop")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "pop"))
    for item in cols:
        print(item)
    print()

    # pop 2sig (time U amp)
    cols = datadf.columns
    cols = [_ for _ in cols if not _.startswith("indi")]
    cols = [_ for _ in cols if "_Sig" in _]
    df = datadf[cols].copy()
    cols = [_.replace("pop_", "pop2sig_") for _ in cols]
    cols = [_.replace("_Sig", "") for _ in cols]
    df.columns = cols
    if do_save:
        df.to_hdf(data_savename, "pop2sig")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "pop2sig"))
    for item in cols:
        print(item)
    print()

    # # pop nSig
    # cols = datadf.columns
    # cols = [_ for _ in cols if not _.startswith("indi")]
    # cols = [_ for _ in cols if "_NSig" in _]
    # if do_save:
    #     datadf[cols].to_hdf(data_savename, "popNsig")
    # print('='*20, '{}({})'.format(os.path.basename(data_savename), 'popNsig'))
    # for item in cols:
    #     print(item)
    # print()

    # pop 3sig (time U amp U fill)
    keys = {
        "_cf": "_centrifugal",
        "_cp": "_centripetal",
        "_ctr": "_centerOnly",
        "_cx": "_cross",
        "_rd": "_rnd",
    }
    cols = sig3df.columns
    for k, v in keys.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = ["pop3sig_" + _ for _ in cols]
    df = sig3df.copy()
    df.columns = cols
    if do_save:
        df.to_hdf(data_savename, "popsig3")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "pop3sig"))
    for item in cols:
        print(item)
    print()


if not "fig2_df" in dir():
    fig2_df, fig2_cols = ldat.load2("new")
if "sig3_df" not in dir():
    sig3_df = get_sig3_df()

save_fig8_data(fig2_df, sig3_df, False)

#%%
plt.close("all")


def sort_stat(age="new"):
    """
    extract description of a the measures
    -> return a pandas dataframe
    """
    if age == "old":
        filename = "data/old/fig2cells.xlsx"
        print("old file fig2cells.xlsx")
        data = pd.read_excel(filename)
        rename_dict = {
            "popVmscpIsolatg": "cpisosect_time50",
            "lagIndiSig": "cpisosect_time50_sig",
            "popVmscpIsoAmpg": "cpisosect_gain50",
            "ampIndiSig": "cpisosect_gain50_sig",
        }
        data.rename(columns=rename_dict, inplace=True)
    elif age == "new":
        latGain50_v_df = ldat.load_cell_contributions(rec="vm", amp="gain", age="new")
        cols = latGain50_v_df.columns
        data = latGain50_v_df[[item for item in cols if "cpisosect" in item]]
    else:
        print("fig2cells.xlsx to be updated")
        return
    # simplification
    simple = False
    if simple:
        cols = [col.replace("cpisosect_", "") for col in data.columns]
        data.columns = cols

    data_col = [_ for _ in data.columns if "_sig" not in _]

    res = pd.DataFrame(index=data_col)
    res["cells"] = [len(data)] * len(data_col)  # nb of cells
    res["mean_all"] = data[data_col].mean().tolist()
    res["std_all"] = data[data_col].std().tolist()
    res["sem_all"] = data[data_col].sem().tolist()

    ares = {"sig_cells": [], "mean_sig": [], "std_sig": [], "sem_sig": []}
    for col in data_col:
        ares["sig_cells"].append(len(data.loc[data[col + "_sig"] == 1, [col]]))
        ares["mean_sig"].append(data.loc[data[col + "_sig"] == 1, [col]].mean()[0])
        ares["std_sig"].append(data.loc[data[col + "_sig"] == 1, [col]].std()[0])
        ares["sem_sig"].append(data.loc[data[col + "_sig"] == 1, [col]].sem()[0])
    for k, v in ares.items():
        res[k] = v

    pd.options.display.float_format = "{:,.2f}".format
    print(res.T)
    return res.T


def plot_stat():
    """
    single plot of the basic parameters obtained for cp_iso
    comparision before and after new treatment
    """
    fig = plt.figure()
    axes = fig.subplots(nrows=1, ncols=2)
    for i, age in enumerate(["old", "new"]):
        tab = table(
            axes[i],
            sort_stat(age).round(2),
            loc="center",
            cellLoc="center",
            edges="L",
            colLabels=["", age, ""],
        )
        tab.set_fontsize(11)
        for j, cell in enumerate(tab.get_celld().values()):
            if j > 17:
                cell.visible_edges = ""
        axes[i].axis("off")
        axes[i].text(
            0.5, 0.8, age + " vm values", horizontalalignment="center", fontsize=14
        )
    fig.tight_layout()
    return fig


descr_df = sort_stat("new")
plot_stat()
#%%
# moved fig3 to popTraces.py
#%%
plt.close("all")


def plot_speed(substract=False):
    """ ex fig 4 """
    filename = "data/data_to_use/fig4.xlsx"
    df = pd.read_excel(filename, engine="openpyxl")
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    # OBSERVATION bottom raw 0 baseline has been decentered by police and ticks size changes
    df.index = df.index - middle
    df.index = df.index / 10
    cols = ["centerOnly", "100%", "70%", "50%", "30%"]
    df.columns = cols
    colors = [
        "k",
        std_colors["red"],
        speed_colors["dark_orange"],
        speed_colors["orange"],
        speed_colors["yellow"],
    ]
    alphas = [0.8, 1, 0.8, 0.8, 1]
    if substract:
        ref = df.centerOnly
        df = df.subtract(ref, axis=0)
        # stack
        # stacks = []
        # for i, col in enumerate(df.columns[:-5:-1]):
        #     df[col] += i / 10 * 2
        #     stack.append(i / 10 * 2)
    fig = plt.figure(figsize=(8.5, 5.5))
    # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=1.5)
    ax.set_ylabel("Normalized Vm")
    # ax.text(0.8, 0.9, 'CP-ISO', ha='left', va='center',
    #         transform=ax.transAxes, size='large')

    # fontname = 'Arial', fontsize = 14)
    ax.set_xlabel("Relative Time (ms)")
    for ax in fig.get_axes():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()
    # fig.legend()
    # old xlims ax.set_xlim(-40, 45)
    ax.set_xlim(-90, 65)

    ax.set_ylim(-0.15, 1.2)
    # ax.axvline(0, alpha=0.2, color='k')
    ax.axhline(0, alpha=0.2, color="k")
    # old custom_ticks = np.linspace(-40, 40, 5)
    # custom_ticks = np.linspace(-90, 65, 20)
    # ax.set_xticks(custom_ticks)

    # leg = ax.legend(loc='upper left', markerscale=None, frameon=False,
    #                handlelength=0)
    # for line, text in zip(leg.get_lines(), leg.get_texts()):
    #    text.set_color(line.get_color())
    txt = "CP-ISO \nn=12"
    ax.text(
        0.1, 0.8, txt, ha="center", va="center", transform=ax.transAxes, size="large"
    )

    # ax.annotate("n=12", xy=(0.1, 0.8), xycoords="axes fraction", ha='center')
    # bluePoint
    x = 0
    y = df.loc[0]["centerOnly"]
    vspread = 0.06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color="tab:gray")
    ax.axvline(x, linewidth=2, color="tab:blue", linestyle=":")
    # ax.plot(0, , 'o', color=std_colors['blue'],
    #         ms=10, alpha=0.8)
    if substract:
        ax.set_ylim(-0.05, 0.4)
        custom_ticks = np.linspace(0, 0.3, 4)
        ax.set_yticks(custom_ticks)
        # old xlims ax.set_xlim(-80, 60)
        ax.set_xlim(-90, 65)
    else:
        custom_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "centrifigs.py:plot_speed", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        fig.text(0.5, 0.01, "speed", ha="center", va="bottom", alpha=0.4)
    return fig


fig = plot_speed()

save = False
if save:
    file = "f10_speed"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        file_name = os.path.join(paths["save"], (file + ext))
        fig.savefig(file_name)


def save_fig11_data(do_save=False):

    conds, key_dico = config.std_names()

    filename = "data/data_to_use/fig4.xlsx"
    df = pd.read_excel(filename, engine="openpyxl")
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10

    cols = df.columns
    cols = ["_" + _ + "_" for _ in cols]
    for k, v in conds:
        cols = [_.replace(k, v) for _ in cols]
    for k, v in key_dico.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = [_.strip("_") for _ in cols]
    df.columns = cols

    data_savename = os.path.join(paths["figdata"], "fig11.hdf")
    if do_save:
        df.to_hdf(data_savename, "speed")
        print("=" * 20, "{}({})".format(os.path.basename(data_savename), "speed"))
        for item in cols:
            print(item)
        print()


save_fig11_data(True)
#%% test to change the x scale


def adjust_scale(figlist, lims):
    for fig in [fig1, fig2]:
        ax = fig.get_axes()[0]
        ax.set_xlim(lims)
        #        left = list(np.arange(lims[0], 0, 25))[1:]
        left = list(np.arange(0, lims[0], -25))[1:][::-1]
        right = list(np.arange(0, lims[1], 25))
        custom_ticks = left + right
        ax.set_xticks(custom_ticks)


fig_list = [fig1, fig2]
lims = (-40, 45)
lims = (-65, 65)
lims = (-120, 65)
lims = (-160, 65)
lims = (-200, 65)
lims = (-300, 65)

adjust_scale(fig_list, lims)

#%% fig 5 <-> sup 7

plt.close("all")


def plot_highLowSpeed():
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    #    filenames = ['data/figSup7a.xlsx',
    # 'data/figSup5bis.xlsx']
    #'data/figSup7b.xlsx']
    filenames = ["data/data_to_use/highspeed.xlsx", "data/data_to_use/lowspeed.xlsx"]
    titles = ["High speed", "Low speed"]

    filename = filenames[0]
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10
    # reduce the time range
    df = df.loc[-100:300]
    # remove the negative values
    for col in df.columns:
        df[col].loc[df[col] < 0] = 0
    cols = ["scp-Iso-Stc-HighSpeed", "scp-Cross-Stc-HighSpeed"]  # ,
    # 'scp-Cross-Stc-LowSpeed', 'scp-Iso-Stc-LowSpeed']
    df.columns = cols
    colors = [std_colors["red"], std_colors["yellow"]]
    darkcolors = [std_colors["dark_red"], std_colors["dark_yellow"]]
    alphas = [0.7, 0.7]

    fig = plt.figure(figsize=(4, 7))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.fill_between(
            df.index,
            df[col],
            facecolor=colors[i],
            edgecolor="black",
            alpha=1,
            linewidth=1,
        )
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_ylim(0, 5.5)

    filename = filenames[1]
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10
    # reduce the time range
    df = df.loc[-100:300]
    # remove the negative values
    for col in df.columns:
        df[col].loc[df[col] < 0] = 0

    cols = ["scp-Cross-Stc-LowSpeed", "scp-Iso-Stc-LowSpeed"]
    df.columns = cols
    colors = colors[::-1]
    darkcolors = darkcolors[::-1]
    ax2 = fig.add_subplot(212)
    for i, col in enumerate(cols[:2]):
        ax2.fill_between(
            df.index,
            df[col],
            facecolor=colors[i],
            edgecolor="black",
            alpha=1,
            linewidth=1,
        )
    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines["bottom"].set_visible(True)
    ax2.set_ylim(0, 11.5)
    ax2.set_xlabel("Time (ms)")

    ax1.annotate("100°/s", xy=(0.2, 0.95), xycoords="axes fraction", ha="center")

    ax2.annotate("5°/s", xy=(0.2, 0.95), xycoords="axes fraction", ha="center")

    for ax in fig.get_axes():
        ax.axvline(0, alpha=0.1, color="k")
        ax.axhline(0, alpha=0.1, color="k")
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()
    fig.text(0.02, 0.5, "Firing rate (spk/s)", va="center", rotation="vertical")
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_highLowSpeed",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


fig = plot_highLowSpeed()
#%%


#%% fig 9
plt.close("all")


def plot_figure9CD(data, colsdict):
    """
    plot_figure9CD
    """
    colors = ["k", std_colors["red"]]
    alphas = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(11.6, 5))  # fig = plt.figure(figsize=(8, 8))
    ax0 = fig.add_subplot(1, 2, 1)
    cols = colsdict["popVmSig"]
    # ax.set_title('significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax0.plot(df[col], color=colors[i], alpha=alphas[i], label=col)
        # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax0.fill_between(
            df.index, df[col[0]], df[col[1]], color=colors[i], alpha=alphas[i] / 2
        )
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    # ax.plot(x0, y, 'o', color=std_colors['blue'], ms=10, alpha=0.8)
    # ax.plot(x1, y, '|', color=std_colors['blue'], ms=10, alpha=0.8)
    # ax.axhline(y, x1, x0, color=std_colors['blue'], ms=10, alpha=0.8)
    # ax.annotate("n=10", xy=(0.2, 0.8),
    #               xycoords="axes fraction", ha='center')
    ylabel = "Normalized membrane potential"
    ax0.set_ylabel(ylabel)
    ax0.set_ylim(-0.10, 1.2)
    ax0.set_xlabel("Relative time (ms)")
    ax0.axvline(0, alpha=0.3, color="k")
    lims = ax0.get_xlim()
    ax0.axhline(0, alpha=0.3, color="k")
    # lims = ax1.get_ylim()
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax0.set_yticks(custom_ticks)

    # hist
    ax1 = fig.add_subplot(1, 2, 2)
    #    filename = 'data/fig2cells.xlsx'
    filename = "data/old/fig2cells.xlsx"
    print(">>>>> beware old file fig2cells <<<<<")
    df = pd.read_excel(filename)
    # cols = df.columns[:2]
    # signs = df.columns[2:]
    df.index += 1  # cells = 1 to 37

    nsig = df.loc[df.lagIndiSig == 0].popVmscpIsolatg.tolist()
    sig = df.loc[df.lagIndiSig == 1].popVmscpIsolatg.tolist()

    bins = np.arange(-5, 36, 5) - 2.5
    ax1.hist(
        [sig, nsig],
        bins=bins,
        stacked=True,
        color=[std_colors["red"], "None"],
        edgecolor="k",
        linewidth=1,
    )
    ax1.set_xlim(-10, 35)
    # adjust nb of ticks
    lims = ax1.get_ylim()
    custom_ticks = np.linspace(lims[0], 18, 7, dtype=int)
    # custom_ticks = np.arange(0, 13, 4)
    ax1.set_yticks(custom_ticks)
    ax1.set_yticklabels(custom_ticks)
    ax1.axvline(0, linestyle="--", color="k")
    ax1.set_ylabel("Number of cells")
    ax1.set_xlabel(r"$\Delta$ Phase (ms)")

    for ax in fig.get_axes():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    # remove the space between plots
    # fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_figure9CD",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


# fig2_df, fig2_cols = load2()
plot_figure9CD(fig2_df, fig2_cols)


#%%
plt.close("all")


def plot_figSup2B(kind="pop", age="new"):
    """
    plot supplementary figure 1 : Vm with random Sector control
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    if age == "old":
        filenames = {
            "pop": "data/old/figSup3.xlsx",
            "sig": "data/old/figSup3bis.xlsx",
            "nonsig": "data/old/figSup3bis2.xlsx",
        }
        print(">>>>> files figSup3.xlsx ... should be updated <<<<<")
    else:
        print("figSup3.xls file should be updated")
    titles = {
        "pop": "all cells",
        "sig": "individually significant cells",
        "nonsig": "individually non significants cells",
    }
    # samplesize
    cellnumbers = {"pop": 37, "sig": 10, "nonsig": 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10
    cols = [
        "CENTER-ONLY",
        "CP-ISO",
        "CF-ISO",
        "CP-CROSS",
        "RND-ISO SECTOR",
        "RND-ISO-FULL",
    ]
    df.columns = cols
    colors = [std_colors[item] for item in "k red green yellow blue blue".split()]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(8, 7))
    # SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    if anot:
        fig.suptitle("Vm " + titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        if i == 4:
            ax.plot(
                df[col],
                linestyle="dotted",
                color=colors[i],
                alpha=alphas[i],
                label=col,
                linewidth=2,
            )
        elif i != 4:
            ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
    ax.set_ylabel("Normalized membrane potential")
    ax.set_xlabel("Relative time (ms)")
    for ax in fig.get_axes():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    ax.axhline(0, alpha=0.3, color="k")
    ax.axvline(0, alpha=0.3, color="k")
    ax.set_ylim(-0.1, 1)
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    # blue point
    ax.plot(
        0, df.loc[0]["CENTER-ONLY"], "o", color=std_colors["blue"], ms=10, alpha=0.8
    )

    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False, handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_figSup2B",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


fig = plot_figSup2B("pop", age="old")
# fig = plot_figSup2B('sig')
# fig = plot_figSup2B('nonsig')


#%%
plt.close("all")


def plot_figSup4(kind, overlap=True):
    """
    plot supplementary figure 3:
        Vm all conditions of surround-only stimulation CP-ISO sig
    input : kind in ['minus'', 'plus]
        'minus': Surround-then-center - Center Only Vs Surround-Only,
        'plus': Surround-Only + Center only Vs Surround-then-center]
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

    # filenames = {'minus' : 'data/figSup2m.xlsx',
    #              'plus' : 'data/figSup2p.xlsx'}
    filenames = {
        "minus": "data/data_to_use/condfillminus.xlsx",
        "plus": "data/data_to_use/condfillplus.xlsx",
    }

    titles = {
        "minus": "Surround-then-center minus center only",
        "plus": "Surround-only plus center-only",
    }

    yliminf = {"minus": -0.15, "plus": -0.08}
    ylimsup = {"minus": 0.4, "plus": 1.14}

    # samplesize
    # cellnumbers = {'minus' : 12, 'plus': 12}
    # ncells = cellnumbers[kind]
    # ylimtinf = yliminf[kind]
    # ylimtsup = ylimsup[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    # adjust time
    df.index = df.index / 10
    df = df.loc[-150:150]
    # rename
    cols = [
        "CP_Iso_Stc",
        "CP_Iso_Stc_SeUp",
        "CP_Iso_Stc_SeDw",
        "CP_Iso_Stc_Dlp",
        "CF_Iso_Stc",
        "CF_Iso_Stc_SeUp",
        "CF_Iso_Stc_SeDw",
        "CF_Iso_Stc_Dlp",
        "CP_Cross_Stc",
        "CP_Cross_Stc_SeUp",
        "CP_Cross_Stc_SeDw",
        "CP_Cross_Stc_Dlp",
        "RND_Iso_Stc_Sec",
        "RND_Iso_Stc_SeUp_Sec",
        "RND_Iso_Stc_SeDw_Sec",
        "RND_Iso_Stc_Dlp_Sec",
        "RND_Iso_Stc_Full",
        "RND_Iso_Stc_SeUp_Full",
        "RND_Iso_Stc_SeDw_Full",
        "RND_Iso_Stc_Dlp_Full",
    ]
    df.columns = cols
    # colors
    light_colors = [std_colors[item] for item in "red green yellow blue blue".split()]
    dark_colors = [
        std_colors[item]
        for item in "dark_red dark_green dark_yellow dark_blue dark_blue ".split()
    ]
    alphas = [0.7, 0.2]  # front, fillbetween
    # traces -> lists of 4 columns ie each condition (val, up, down, sum)
    col_seg = [cols[i : i + 4] for i in np.arange(0, 17, 4)]

    fig = plt.figure(figsize=(4, 8))
    for i in range(5):
        if i == 0:
            ax = fig.add_subplot(5, 1, i + 1)
        else:
            ax = fig.add_subplot(5, 1, i + 1, sharex=ax, sharey=ax)
        toPlot = col_seg[i]
        col = toPlot[0]
        # print(col)
        ax.plot(
            df[col], color=light_colors[i], alpha=alphas[0], label=col, linewidth=1.5
        )
        ax.fill_between(
            df.index,
            df[toPlot[1]],
            df[toPlot[2]],
            color=light_colors[i],
            alpha=alphas[1],
        )
        col = toPlot[-1]
        ax.plot(
            df[col], color=dark_colors[i], alpha=alphas[0], label=col, linewidth=1.5
        )
        # ax.axvspan(xmin=0, xmax=50, ymin=0.27, ymax=0.96,
        #            color='grey', alpha=alphas[1])

        ax.spines["top"].set_visible(False)
        ax.set_facecolor("None")
        # axis on both sides
        # set_ticks_both(ax.yaxis)
        # if overlap:
        #     #label left:
        #     if i % 2 == 0:
        #         ax.spines['right'].set_visible(False)
        #     #label right
        #     else:
        #         ax.spines['left'].set_visible(False)
        #         ax.yaxis.tick_right()
        # else:
        #     ax.spines['right'].set_visible(False)
        if i != 4:
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel("Relative time (ms)")
    for i, ax in enumerate(fig.get_axes()):
        ax.axhline(0, alpha=0.3, color="k")
        custom_ticks = np.arange(0, 0.3, 0.1)
        ax.set_yticks(custom_ticks)
        for spine in ["left", "right"]:
            ax.spines[spine].set_visible(False)
        ax.vlines(ax.get_xlim()[0], 0, 0.2)
    for ax in fig.get_axes():
        lims = ax.get_ylim()
        print(lims)
        r1 = patches.Rectangle(
            (0, 0), 50, 0.4, color="grey", alpha=0.2  # ax.get_ylim()[1]
        )
        ax.add_patch(r1)

    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.3, wspace=0.2)

    fig.text(
        0.01, 0.5, "Normalized membrane potential", va="center", rotation="vertical"
    )

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "centrifigs.py:plot_figSup4", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


fig = plot_figSup4("minus", overlap=True)
# fig = plot_figSup4('plus')
# pop all cells
#%%
plt.close("all")


def plot_fig_sup3B(kind, stimmode, age="new"):
    """
    plot supplementary figure 5: All conditions spiking responses of Sector and Full stimulations
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    if age == "old":
        filenames = {"pop": "data/old/figSup5Spk.xlsx"}  # ,
        #'sig': 'data/old/figSup1bis.xlsx',
        #'nonsig': 'data/old/figSup1bis2.xlsx'}
        print(">>>>> beware : old file <<<<<")
    else:
        print("figSup5Spk should be updated")
        return
    titles = {"pop": "all cells"}  # ,
    #'sig': 'individually significant cells',
    #'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {"pop": 20}  # , 'sig': x1, 'nonsig': x2}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = (df.index - middle) / 10
    cols = [
        "CENTER-ONLY-SEC",
        "CP-ISO-SEC",
        "CF-ISO-SEC",
        "CP-CROSS-SEC",
        "RND-ISO-SEC",
        "CENTER-ONLY-FULL",
        "CP-ISO-FULL",
        "CF-ISO-FULL",
        "CP-CROSS-FULL",
        "RND-ISO-FULL",
    ]
    df.columns = cols
    colors = [std_colors[item] for item in "k red green yellow blue".split()]
    # alphas = [0.5, 0.8, 0.5, 1, 0.6]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(6.5, 5.5))
    # SUGGESTION: make y dimension much larger to see maximize visual difference
    # between traces
    if anot:
        fig.suptitle(titles[kind] + " spikes")
    ax = fig.add_subplot(111)
    if stimmode == "sec":
        for i, col in enumerate(cols[:5]):
            ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
    else:
        if stimmode == "ful":
            for i, col in enumerate(cols[5:]):
                ax.plot(
                    df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2
                )

    ax.set_ylabel("Normalized firing rate")
    ax.set_xlabel("Relative time (ms)")
    for ax in fig.get_axes():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    ax.axhline(0, alpha=0.2, color="k")
    ax.set_ylim(-0.2, 1.1)
    ax.axvline(0, alpha=0.2, color="k")
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    # bluePoint
    ax.plot(0, df.loc[0]["CENTER-ONLY-FULL"], "o", color=colors[-1], ms=10, alpha=0.8)

    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False, handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #             xycoords="axes fraction", ha='center')
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_fig_sup3B",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    return fig


# fig = plot_fig_sup3B('pop', 'sec',  age='old')
fig = plot_fig_sup3B("pop", "ful", age="old")

#%%
plt.close("all")


def plot_figSup6(kind, age="new"):
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population,
            'sig': individually significants cells,
            'nonsig': non significant cells]
    """
    if age == "old":
        filenames = {"pop": "data/old/figSup6.xlsx"}  # ,
        # 'sig': 'data/old/figSup1bis.xlsx',
        # 'nonsig': 'data/old/figSup1bis2.xlsx'}
        print(" >>>>> beware : old file <<<<< ")
    else:
        print("figSup6.xlsx should be updated")
        return
    titles = {"pop": "all cells"}  # ,1
    # 'sig': 'individually significant cells',
    # 'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {"pop": 37}  # , 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10
    cols = ["CENTER-ONLY", "CP-ISO", "CF-ISO", "CP-CROSS", "RND-ISO"]
    df.columns = cols
    colors = [
        "k",
        std_colors["red"],
        std_colors["green"],
        std_colors["yellow"],
        std_colors["blue"],
    ]
    colors = [std_colors[item] for item in "k red green yellow blue".split()]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8]

    fig = plt.figure(figsize=(6, 10))
    # fig.suptitle(titles[kind])
    # fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True, figsize = (8,7))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols):
        if i in (0, 1, 4):
            ax1.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    # ax1.set_ylabel('Normalized membrane potential')
    ax1.set_ylim(-0.2, 1.1)

    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        if i in (0, 1, 3):
            ax2.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)

    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines["bottom"].set_visible(True)
    # ax2.set_ylabel('Normalized membrane potential')
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_xlabel("Relative time (ms)")

    # axes = list(fig.get_axes())
    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    #     leg = ax.legend(loc=2, markerscale=None, frameon=False,
    #                     handlelength=0)
    #     for line, text in zip(leg.get_lines(), leg.get_texts()):
    #         text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #             xycoords="axes fraction", ha='center')

    for ax in fig.get_axes():
        ax.set_xlim(-15, 30)
        ax.set_ylim(-0.2, 1.1)
        ax.axvline(0, alpha=0.1, color="k")
        ax.axhline(0, alpha=0.1, color="k")
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()
    fig.text(
        -0.04,
        0.6,
        "Normalized membrane potential",
        fontsize=16,
        va="center",
        rotation="vertical",
    )
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "centrifigs.py:plot_figSup6", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


fig = plot_figSup6("pop", age="old")

#%%
plt.close("all")


def plot_sorted_responses(dico):
    """
    plot the sorted cell responses
    input = conditions parameters

    """
    cols = [std_colors[item] for item in "red green yellow blue".split()]
    # duplicate for two columns
    colors = []
    for item in zip(cols, cols):
        colors.extend(item)  # data (call)
    # TODO adapty for vm, engy, ...
    # load_cell_contributions(rec='vm', amp='gain', age='new')
    df = ldat.load_cell_contributions(rec=dico["kind"], amp=dico["amp"])
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if dico["spread"] in item]
    # filter -> only significative cells
    traces = [item for item in traces if "sig" not in item]
    # text labels
    title = dico["kind"] + " (" + dico["spread"] + ")"
    # title = title_dico[dico['kind']]
    anotx = "Cell rank"
    anoty = [r"$\Delta$ phase (ms)", r"$\Delta$ amplitude"]
    # (fraction of Center-only response)']
    if dico["amp"] == "engy":
        anoty[1] = "energy"
    # plot
    fig, axes = plt.subplots(
        4, 2, figsize=(12, 16), sharex=True, sharey="col", squeeze=False
    )  # •sharey=True,
    fig.suptitle(title)
    axes = axes.flatten()
    x = range(1, len(df) + 1)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + "_sig"
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0: "w", 1: edge_color}
        select = df[[name, sig_name]].sort_values(by=[name, sig_name], ascending=False)
        bar_colors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        ax.set_title(name)
        ax.bar(
            x,
            select[name],
            color=bar_colors,
            edgecolor=edge_color,
            alpha=0.8,
            width=0.8,
        )
        if i in [0, 1]:
            ax.set_title(anoty[i])
    for i, ax in enumerate(axes):
        ax.ticklabel_format(useOffset=True)
        ax.axhline(0, alpha=0.3, linestyle=":", color="k")
        for spine in ["left", "top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        if i in range(6):
            ax.xaxis.set_visible(False)
        else:
            ax.set_xlabel(anotx)
            ax.xaxis.set_label_coords(0.5, -0.025)
            ax.set_xticks([1, len(df)])
            ax.set_xlim(0, len(df) + 2)
    # left
    for ax in axes[::2]:
        ax.vlines(0, 0, 20, color="k", linewidth=2)
        custom_ticks = np.linspace(0, 20, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    # right
    for ax in axes[1::2]:
        ax.vlines(0, 0, 1, color="k", linewidth=2)
        custom_ticks = np.linspace(0, 1, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    # align each row yaxis on zero between subplots
    gfunc.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gfunc.change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots
    fig.subplots_adjust(hspace=0.00, wspace=0.00)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_sorted_responses",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    return fig


parameter_dico = {
    "kind": "vm",
    "spread": "sect",
    "position": "cp",
    "theta": "cross",
    "extra": "stc",
    "amp": "engy",
}

fig = plot_sorted_responses(parameter_dico)

#%%
plt.close("all")

# iterate through conditions for plotting
for amp in ["gain", "engy"]:
    parameter_dico["amp"] = amp
    for kind in ["vm", "spk"]:
        parameter_dico["kind"] = kind
        for spread in ["sect", "full"]:
            parameter_dico["spread"] = spread
            fig = plot_sorted_responses(parameter_dico)

#%% opt


def plot_speed_multigraph(df, speedcolors):
    """
    plot the speed effect of centrigabor protocol
    """
    colors = [speed_colors[item] for item in "k red dark_orange orange yellow".split()]
    alphas = [0.8, 1, 0.8, 0.8, 1]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Aligned on Center-Only stimulus onset (t=0 ms)")
    # build grid
    gs = fig.add_gridspec(5, 2)
    left_axes = []
    left_axes.append(fig.add_subplot(gs[4, 0]))
    for i in range(4):
        left_axes.append(fig.add_subplot(gs[i, 0]))
    right_ax = fig.add_subplot(gs[:, 1])
    # to identify the plots (uncomment to use)
    for i, ax in enumerate(left_axes):
        st = str("ax {}".format(i))
        ax.annotate(st, (0.5, 0.5))
        # ax.set_xtickslabels('', minor=False)
    # (manipulate the left_axes list to reorder the plots if required)
    # axes.set_xticklabels(labels, fontdict=None, minor=False)
    # plot left
    # axes = axes[1:].append(axes[0])   # ctrl at the bottom
    cols = df.columns
    for i, ax in enumerate(left_axes):
        ax.plot(
            df.loc[-140:40, [cols[i]]],
            color="black",
            scalex=False,
            scaley=False,
            label=cols[i],
        )
        ax.fill_between(df.index, df[cols[i]], color=colors[i])
        ax.yaxis.set_ticks([0, 0.1])
        ax.set_xlim(-140, 40)
        ax.set_ylim(-0.15, 0.25)
    # add labels
    left_axes[3].set_ylabel("Normalized Membrane potential")
    left_axes[0].set_xlabel("Relative time to center-only onset (ms)")
    left_axes[0].xaxis.set_ticks(np.arange(-140, 41, 40))
    ticks = np.arange(-140, 41, 20)
    for i, ax in enumerate(left_axes[1:]):
        ax.set_xticks(ticks, minor=False)
        ax.tick_params(axis="x", labelsize=0)

    # plot right
    for i, col in enumerate(df.columns):
        right_ax.plot(
            df.loc[40:100, [col]], color=colors[i], label=col, alpha=alphas[i]
        )
        maxi = float(df.loc[30:200, [col]].max())
        right_ax.axhline(maxi, 40, 50, color=colors[i])
    right_ax.set_xlabel("Relative time to center-only onset (ms)")
    # adjust
    for ax in fig.get_axes():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in left_axes:
        ax.axvline(0, alpha=0.5, color="k")
    # adjust spacing
    gs.update(wspace=0.2, hspace=0.05)
    # add ticks to the top
    right_ax.tick_params(axis="x", bottom=True, top=True)
    # legend
    # leg = right_ax.legend(loc='lower right', markerscale=None,
    #                       handlelength=0, framealpha=1)
    # for line, text in zip(leg.get_lines(), leg.get_texts()):
    #     text.set_color(line.get_color())
    # fig.tight_layout()
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_speed_multigraph",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


df = pd.read_excel("data/data_to_use/speedt0.xlsx")
df.set_index("time", inplace=True)
fig = plot_speed_multigraph(df, speed_colors)

#%% test to analyse with x(t) = x(t) - x(t-1)


def plot_speeddiff():
    """ speed diff """
    colors = [speed_colors[item] for item in "k red dark_orange orange yellow".split()]
    alphas = [0.5, 1, 0.8, 0.8, 1]

    df = pd.read_excel("data/data_to_use/speedt0.xlsx")
    df.set_index("time", inplace=True)
    # perform shift (x(t) <- x[t) - x(t-1]
    for col in df.columns:
        df[col] = df[col] - df[col].shift(1)

    fig = plt.figure()
    title = "speed, y(t) <- y(t) - y(t-1), only positives values"
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    cols = df.columns.to_list()
    cols = cols[::-1]
    for j, col in enumerate(cols):
        i = len(cols) - j - 1
        print(i, j)
        xvals = df.loc[-140:100].index
        yvals = df.loc[-140:100, [cols[i]]].values[:, 0]
        # replace negative values <-> negative slope by 0
        yvals = yvals.clip(0)
        ax.fill_between(
            xvals,
            yvals + i / 400,
            i / 400,
            color=colors[j],
            label=cols[i],
            alpha=alphas[j],
        )
    ax.axvline(0, alpha=0.2, color="k")
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    fig.legend()
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "centrifigs.py:plot_speeddiff",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


fig = plot_speeddiff()
