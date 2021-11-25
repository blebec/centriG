#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:21:38 2021

@author: cdesbois
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

os.chdir(paths["pg"])

#%%
# age = ["old", "new"][1]
# if "fig2_df" not in globals():
#     fig2_df, fig2_cols = ldat.load2(age)


def load_fig8_cpIsoGain_initial(printTraces=False):
    filename = os.path.join(paths["pg"], "data", "data_to_use", "fig2_2traces.xlsx")
    inidf = pd.read_excel(filename, engine="openpyxl")
    # centering
    middle = (inidf.index.max() - inidf.index.min()) / 2
    inidf.index = (inidf.index - middle) / 10
    inidf = inidf.loc[-200:150]
    cols = inidf.columns
    cols = [_.casefold() for _ in cols]
    cols = [_.replace("vm", "_vm_") for _ in cols]
    cols = [_.replace("spk", "_spk_") for _ in cols]
    cols = [_.replace("scpiso", "_s_cpiso_") for _ in cols]
    cols = [_.replace("ctr", "_ctr_") for _ in cols]
    cols = [_.replace("nsig", "_nsig_") for _ in cols]
    cols = [_.replace("sig", "_sig_") for _ in cols]
    cols = [_.replace("n_sig", "nsig") for _ in cols]
    cols = [_.replace("_stc", "_stc_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    cols = [_.strip("_") for _ in cols]
    cols = [_.replace("_s_", "_") for _ in cols]
    cols = [_.replace("pop_", "popN2sig_") if "_nsig" in _ else _ for _ in cols]
    cols = [_.replace("_nsig", "") for _ in cols]
    cols = [_.replace("pop_", "pop2sig_") if "_sig" in _ else _ for _ in cols]
    cols = [_.replace("_sig", "") for _ in cols]
    inidf.columns = cols

    print("fig8_cpIsoGain_initial")
    if printTraces:
        for col in inidf.columns:
            print(col)
        print()

    aset = set()
    for _ in cols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))
    return inidf


def load_fig8_cpIsoGain_sup(printTraces=False):
    supfilename = os.path.join(paths["sup"], "fig8_supdata.xlsx")
    supdf = pd.read_excel(supfilename, keep_default_na=True, na_values="")
    # centering
    middle = (supdf.index.max() - supdf.index.min()) / 2
    supdf.index = (supdf.index - middle) / 10
    supdf = supdf.loc[-200:150]
    scols = supdf.columns
    scols = [_.casefold() for _ in scols]
    scols = [_.replace("ind_cell_", "indi_") for _ in scols]
    scols = [_.replace("indi_", "indivm_") if "_vm_" in _ else _ for _ in scols]
    scols = [_.replace("indi_", "indispk_") if "_spk_" in _ else _ for _ in scols]

    scols = [_.replace("pop_37_", "pop_") for _ in scols]
    scols = [_.replace("pop_22_", "pop_") for _ in scols]
    scols = [_.replace("pop_15_", "pop2sig_") for _ in scols]
    scols = [_.replace("pop_6_", "pop2sig_") for _ in scols]
    scols = [_.replace("pop_20_", "pop3sig_") for _ in scols]
    scols = [_.replace("pop_10_", "pop3sig_") for _ in scols]

    for pop in ["pop", "pop2sig", "pop3sig"]:
        scols = [_.replace(pop + "_", pop + "vm_") if "_vm" in _ else _ for _ in scols]
        scols = [
            _.replace(pop + "_", pop + "spk_") if "_spk" in _ else _ for _ in scols
        ]

    scols = [_.replace("_vm", "") for _ in scols]
    scols = [_.replace("_spk", "") for _ in scols]
    scols = [_.replace("vm_", "_vm_") for _ in scols]
    scols = [_.replace("spk_", "_spk_") for _ in scols]

    scols = [_.replace("ctr_stc_", "ctr_") for _ in scols]
    scols = [_.replace("cpcross", "cpx") for _ in scols]

    supdf.columns = scols

    print("beware : an unamed 38 column exists")

    print("fig8_cpIsoGain_sup")
    if printTraces:
        for col in supdf.columns:
            print(col)
        print()

    aset = set()
    for _ in scols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))

    return supdf


def load_fig8_cpIsoGain_pop3sig(key="sector", printTraces=False):
    """ get sig amp U time U fill data aka sig3"""

    if key != "sector":
        print("{} should be implemented".format(key))
        return
    filename = os.path.join(
        paths["owncFig"],
        "data/averageTraces/controlsFig/union_idx_fill_sig_sector.xlsx",
    )
    sig3df = pd.read_excel(filename, engine="openpyxl")
    cols = gfunc.new_columns_names(sig3df.columns)
    cols = [item.replace("sig_", "pop3sig_") for item in cols]
    cols = [item.replace("full_rd", "frnd") for item in cols]
    cols = [item.replace("sect_rd", "srnd") for item in cols]
    cols = [st.replace("cp_iso", "cpiso") for st in cols]
    cols = [st.replace("cf_iso", "cfiso") for st in cols]
    cols = [st.replace("sect_cx", "cpx") for st in cols]
    cols = [st.replace("_sect_", "_") for st in cols]
    cols = [st.replace("__", "_") for st in cols]
    cols = [st.replace("_.1", "") for st in cols]
    sig3df.columns = cols
    # adjust time scale
    middle = (sig3df.index.max() - sig3df.index.min()) / 2
    sig3df.index = (sig3df.index - middle) / 10

    print("fig8_cpIsoGain_pop3sig")
    if printTraces:
        for col in sig3df.columns:
            print(col)
        print()

    aset = set()
    for _ in cols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))

    return sig3df


ini_df = load_fig8_cpIsoGain_initial()
sig3_df = load_fig8_cpIsoGain_pop3sig()
sup_df = load_fig8_cpIsoGain_sup()

cols = ini_df.columns
inicols = ini_df.columns
[_ for _ in inicols if _.startswith('indi')]
df1 = ini_df[_ for _ in inicols if _.startswith('indi')]
df1 = ini_df[[_ for _ in inicols if _.startswith('indi')]]
sigcols = sig3_df.colums
sigcols = sig3_df.columns
[_ for _ in sigcols if _.startswith('indi')]
supcols = sup_df.columns
[_ for _ in supcols if _.startswith('indi')]
df2 = sup_df[[_ for _ in supcols if _.startswith('indi')]]



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

figure = plot_cpIsoGain(
    datadf=fig2_df.copy(), sig3df=sig3_df, colsdict=fig2_cols, anot=anot, age=age
)
save = False
if save:
    name = "f8_cpIsoGain_alt"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["save"], (name + ext)))
