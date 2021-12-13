#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# from socket import gethostname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyperclip

import config


# def build_paths():
#     """ build a paths dictionary """
#     host = gethostname()
#     paths = {}
#     for key in ['home', 'data', 'ownc', 'save']:
#         paths[key] = ''
#     if host == 'PC-Chris-linux':
#         paths['data'] = '/run/user/1002/gvfs/smb-share:server=157.136.60.11,\
#         share=posteintra/centrigabor_analyses/centrigabor_neurons/\
#         benoit_thesis_latestVersion/Lat50_and_DRES50_txt'
#         paths['home'] = '/mnt/hWin/Chris'
#         paths['ownc'] = os.path.join(paths['home'], 'ownCloud')
#         paths['save'] = os.path.join(paths['home'], 'ownCloud', 'centrGfig')
#     # on mac : paths['data'] = '/Volumes/posteIntra/centrigabor_analyses/\
#     #centrigabor_neurons/benoit_thesis_latestVersion/Lat50_and_DRES50_txt/'
#     return paths

# paths = build_paths()

paths = config.build_paths()
paths["save"] = os.path.join(paths.get("pg"), "dataCorr", "hd_files")

#%%
def load_from_txt_files(dir_path=""):
    """
    load the latency gain, return a pandas dataframe'
    NB the cells data are in the internal server and in separated txt_files
    """
    # list cells (on the server)
    file = open(os.path.join(dir_path, "cell_list.txt"), "r")
    cells = file.read().split("\n")
    file.close()
    # remove empty item
    cells = [cell for cell in cells if len(cell) > 0]
    #    for item in cells:
    #        if item == '':
    #            cells.remove(item)
    # build dataframe
    df = pd.DataFrame(index=cells)
    for file in os.listdir():
        if os.path.isfile(file):
            name = file.split(".")[0]
            if name != "cell_list":
                data = np.loadtxt(file)
                if len(data) == len(cells):
                    df[name] = np.loadtxt(file)
                else:
                    print(name, ": has a different number of cells")
    # rename columns
    ch_colname = {
        "cp_iso_Dlat50_sector": "cpisoSector_x",
        "cp_iso_Dres50_sector": "cpisoSector_y",
        "cf_iso_Dlat50_sector": "cpcrossSector_x",
        "cf_iso_Dres50_sector": "cpcrossSector_y",
        "cp_cross_Dlat50_sector": "cfisoSector_x",
        "cp_cross_Dres50_sector": "cfisoSector_y",
        "rnd_iso_Dlat50_sector": "rndisoSector_x",
        "rnd_iso_Dres50_sector": "rndisoSector_y",
        "cp_iso_Dlat50_full": "cpisoFull_x",
        "cp_iso_Dres50_full": "cpisoFull_y",
        "cf_iso_Dlat50_full": "cpcrossFull_x",
        "cf_iso_Dres50_full": "cpcrossFull_y",
        "cp_cross_Dlat50_full": "cfisoFull_x",
        "cp_cross_Dres50_full": "cfisoFull_y",
        "rnd_iso_Dlat50_full": "rndisoFull_x",
        "rnd_iso_Dres50_full": "rndisoFull_y",
    }
    df.rename(columns=ch_colname, inplace=True)
    df.index.name = "cells"
    df.reset_index()
    return df


# NB cp_iso_Dlat50_full2 : has a different number of cells
#   cp_iso_Dres50_full2 : has a different number of cells

# load from server
# latency_gain_df  = load_from_txt_files(paths['data'])
# load from hdf file
# latency_gain_df = pd.read_hdf(os.path.join(paths['data'], 'script', 'latGain50.hdf'))
latency_gain_df = pd.read_hdf(os.path.join(paths["save"], "latGain50.hdf"))

## write on server
# file = os.path.join(paths['data'], 'script', 'latGain50.hdf')
# latency_gain_df.to_hdf(file, key='data')
# file = os.path.join(paths['data'], 'script', 'latGain50.csv')
# latency_gain_df.to_csv(file, sep='\t')
## write in owncloud folder
# file = os.path.join(paths['save'], 'latGain50.hdf')
# latency_gain_df.to_hdf(file, key='data')
# file = os.path.join(paths['save'], 'latGain50.csv')
# latency_gain_df.to_csv(file, sep='\t')

#%% histogram of the data
plt.close("all")


def plot_histo(df, num=8, kind="x"):
    """display histogramme of advance or latency for all kind of stimulations"""
    # colors = {"rndiso": "b", "cpcross": "gold", "cpiso": "r", "cfiso": "g"}
    stdcolors = config.std_colors()
    colors = {
        "rndiso": stdcolors["blue"],
        "cpcross": stdcolors["yellow"],
        "cpiso": stdcolors["red"],
        "cfiso": stdcolors["green"],
    }

    fig, axs = plt.subplots(
        2, num // 2, sharex=True, sharey=True, tight_layout=True, figsize=(14, 8)
    )
    axs = axs.flatten()
    if kind == "x":
        title = "latency advance (ms) : histogram + median value"
    elif kind == "y":
        title = "amplitude gain (% center test peak) : histogram + median value"
    else:
        title = ""
    fig.suptitle(title)
    # select data
    cols = [_ for _ in df.columns if _[-1] == kind]
    #    cols = []
    #    for col in df.columns:
    #        if kind in col:
    #            cols.append(col)
    sect = sorted([_ for _ in cols if "Sector" in _])
    full = sorted([_ for _ in cols if "Full" in _])
    cols = sect + full
    # plot
    for i, col in enumerate(cols):
        for key in colors:
            if key in col:
                color = colors[key]
        label = col.replace(kind, "")
        medStr = "med= {:.2f}".format(df[col].median())
        ax = axs[i]
        ax.hist(
            df[col], bins=15, density=True, color=color, label=label
        )  # , sharex=ax0)
        ax.annotate(medStr, xy=(0.6, 0.6), xycoords="axes fraction")
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        ax.vlines(
            df[col].median(), lims[0], lims[1], color=color, alpha=0.4, linewidth=2
        )
        ax.legend()
    # remove the frames
    for ax in axs:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
            ax.yaxis.set_visible(False)
    return fig


fig1 = plot_histo(latency_gain_df, num=8, kind="x")
fig2 = plot_histo(latency_gain_df, num=8, kind="y")
#%%
plt.close("all")


def plot_sectorfull(df):
    """
    plot amplitude vs gain for Sector and Full condition
    input : df = dataframe, key in ['Sector', 'Gain']
    output : matplotlib plot
    """
    # list kinds of stimulation
    kinds = []
    for item in df.columns:
        kind = item.split("_")[0]
        if kind not in kinds:
            kinds.append(kind)
        kinds = sorted(kinds)
    # prepare
    stdcolors = config.std_colors()
    colors = {
        "rndiso": stdcolors["blue"],
        "cpcross": stdcolors["yellow"],
        "cpiso": stdcolors["red"],
        "cfiso": stdcolors["green"],
    }
    # plot
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
    for i, key in enumerate(["Sector", "Full"]):
        ax = axes[i]
        title = key + ": amplitude gain function (& median values)"
        ax.set_title(title)
        ylims = (-0.5, 0.8)
        xlims = (-15, 30)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.hlines(0, xlims[0], xlims[1], alpha=0.3)
        ax.vlines(0, ylims[0], ylims[1], alpha=0.3)
        for kind in kinds:
            if key in kind:
                x = kind + "_x"
                y = kind + "_y"
                label = kind.replace(key, "")
                ax.plot(df[x], df[y], "o", label=label, alpha=0.6, color=colors[label])
                ax.hlines(
                    df[y].median(),
                    xlims[0],
                    xlims[1],
                    alpha=0.3,
                    color=colors[label],
                    linewidth=3,
                )
                ax.vlines(
                    df[x].median(),
                    ylims[0],
                    ylims[1],
                    alpha=0.3,
                    color=colors[label],
                    linewidth=3,
                )
        ax.legend()
    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_xlabel("latency advance (ms)")
    axes[0].set_ylabel("amplitude gain (% center test peak)")


plot_sectorfull(latency_gain_df)

#%% plot cell
plt.close("all")


def plot_rank(df, num=8, kind="_x"):
    """
    plot the ranked cells responses for latenct ('_x') or gain ('_y')
    input : dataframe, num=number of plots, kind in ['_x', '_y']
    output : matplotlib figure
    """
    stdcolors = config.std_colors()
    colors = {
        "rndiso": stdcolors["blue"],
        "cpcross": stdcolors["yellow"],
        "cpiso": stdcolors["red"],
        "cfiso": stdcolors["green"],
    }
    fig, axs = plt.subplots(
        2, num // 2, sharex=False, sharey=True, tight_layout=True, figsize=(14, 8)
    )
    axs = axs.flatten()
    if kind == "_x":
        title = "latency advance, ranked cells"
    elif kind == "_y":
        title = "amplitude gain, ranked cells"
    else:
        title = ""
    fig.suptitle(title)
    # select data
    cols = []
    for col in df.columns:
        if kind in col:
            cols.append(col)
    sect = sorted([item for item in cols if "Sector" in item])
    full = sorted([item for item in cols if "Full" in item])
    cols = sect + full
    # plot
    for i, col in enumerate(cols):
        for colorK in colors.keys():
            if colorK in col:
                color = colors[colorK]
        label = col.replace(kind, "")
        # medStr =  "med= {:.2f}".format(df[col].median())
        ax = axs[i]
        sortedDf = df[col].sort_values(ascending=False).reset_index()
        mini = sortedDf[col].abs().min()
        zeroLoc = sortedDf[sortedDf[col].abs() == mini].index[0]
        x = sortedDf.index.values.tolist()
        y = sortedDf[col].values.tolist()
        ax.bar(x, y, color=color, label=label, alpha=0.7)
        lims = ax.get_ylim()
        ax.vlines(zeroLoc, lims[0] / 5, lims[1] / 5, alpha=0.5)
        # nb cells > 0
        nbCell = sortedDf.loc[sortedDf[col] > 0, "cells"].count()
        totalCell = sortedDf["cells"].count()
        textTitle = str(nbCell) + " / " + str(totalCell) + " cells > centerOnly"
        ax.text(0.4, 0.85, textTitle, transform=ax.transAxes)
        # median value
        textTitle = "median: " + str(round(sortedDf[col].median(), 2))
        ax.text(0.4, 0.75, textTitle, transform=ax.transAxes)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1])
        ax.hlines(sortedDf[col].median(), lims[0], lims[1], color=color, alpha=0.5)
        ax.legend()
        for spine in ["top", "left", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    return fig


fig = plot_rank(latency_gain_df, kind="_x")
fig = plot_rank(latency_gain_df, kind="_y")

#%% full vs sector
def list_stim(df):
    """return a list of the kinds of stimulation based one df.columns names"""
    kinds = []
    for item in [item.split("_")[0] for item in df.columns]:
        if item not in kinds:
            kinds.append(item)
    return kinds


def full_sec(alist):
    """return a dico {'sect', 'full'} of all the stimulations"""
    sect = sorted([item for item in alist if "Sector" in item])
    full = sorted([item for item in alist if "Full" in item])
    return {"sec": sect, "full": full}


def build_median_df(stim_dico, df):
    """construct a dataframe containing the median values for all conditions"""
    # build dataFrame
    cols = [item.replace("Sector", "") for item in stim_dico["sec"]]
    colus = [item + "_x" for item in cols]
    for item in cols:
        colus.append(item + "_y")
    adf = pd.DataFrame(index=["sector", "full"])
    for item in colus:
        adf[item] = [np.nan, np.nan]

    # compute median value
    median_dico = dict(df.median())

    # inset in dataFrame
    for item in median_dico.keys():
        one, two = item.split("_")
        if "Sector" in one:
            loca = "sector"
            protoc = one.replace("Sector", "")
        elif "Full" in one:
            loca = "full"
            protoc = one.replace("Full", "")
        else:
            print("loca problem")
        if "x" in two:
            kind = "_x"
        elif "y" in two:
            kind = "_y"
        else:
            print("kind problem")
        id = protoc + kind
        adf.loc[loca, id] = median_dico[item]
    # cpmpute diff
    adf = adf.T
    adf["f-s"] = adf.full - adf.sector
    adf["s-f"] = adf.sector - adf.full
    adf["f-s/s"] = (adf.full - adf.sector) / adf.sector
    adf["s-f/f"] = (adf.sector - adf.full) / adf.full
    adf["axe"] = adf.reset_index()["index"].apply(lambda x: x.split("_")[1])
    adf.reset_index(inplace=True)
    adf.rename(columns={"index": "stim"}, inplace=True)
    adf.loc[adf.stim.str.contains("_x"), ["axe"]] = "latency"
    adf.loc[adf.stim.str.contains("_y"), ["axe"]] = "gain"
    adf.stim = adf.stim.str.replace("_x", "")
    adf.stim = adf.stim.str.replace("_y", "")
    adf.set_index("stim", inplace=True)
    adf.sort_index(inplace=True)
    return adf


stim_list = list_stim(latency_gain_df)
stim_dico = full_sec(stim_list)
median_df = build_median_df(stim_dico, latency_gain_df)

#%% plot
plt.close("all")


def plot_full_vs_sector(df, key="s-f"):
    fig, axs = plt.subplots(
        1, 2, sharex=True, sharey=False, tight_layout=True, figsize=(7, 5)
    )
    if key == "s-f":
        title = "full->sect     (sect - full)"
    elif key == "f-s":
        title = "sect->full     (full - sect)"
    else:
        title = ""
    fig.suptitle(title)
    for i, axe in enumerate(["latency", "gain"]):
        x = df.loc[df.axe == axe, key].index.tolist()
        y = df.loc[df.axe == axe, key].tolist()
        ax = axs[i]
        if axe == "latency":
            axtitle = axe + "Advance"
        else:
            axtitle = axe
        ax.text(0.5, 0.9, axtitle, transform=ax.transAxes)
        ax.bar(x, y, color=["g", "gold", "r", "b"], alpha=0.5, width=1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    #        if i==1:
    #            ax.spines['left'].set_visible(False)
    #            ax.tick_params('y', width=0)
    return fig


def plot_full_vs_sector_ratio(df, key="s-f/f"):
    fig, axs = plt.subplots(
        1, 2, sharex=True, sharey=True, tight_layout=True, figsize=(7, 5)
    )
    if key == "s-f/f":
        title = "ratio full->sect     (sect - full / full)"
    elif key == "f-s/s":
        title = "ratio sect->full     (full - sect / sec)"
    else:
        title = ""
    fig.suptitle(title)
    for i, axe in enumerate(["latency", "gain"]):
        x = df.loc[df.axe == axe, key].index.tolist()
        y = df.loc[df.axe == axe, key].tolist()
        ax = axs[i]
        if axe == "latency":
            axtitle = axe + "Advance"
        else:
            axtitle = axe
        ax.text(0.5, 0.9, axtitle, transform=ax.transAxes)
        ax.bar(x, y, color=["g", "gold", "r", "b"], alpha=0.5, width=1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if i == 1:
            ax.spines["left"].set_visible(False)
            ax.tick_params("y", width=0)
    return fig


plot_full_vs_sector(median_df, "s-f")
plot_full_vs_sector(median_df, "f-s")
plot_full_vs_sector_ratio(median_df, "s-f/f")
plot_full_vs_sector_ratio(median_df, "f-s/s")

from scipy import stats

#%%
from sklearn import linear_model


def regress(df, out=False):
    """compute the correlation between conditions"""
    stat_df = pd.DataFrame(
        index=["slope", "intercept", "r_value", "p_value", "std_err"]
    )
    for item in list_stim(df):
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[[item + "_x", item + "_y"]]
        )
        adico = {}
        adico["slope"] = slope
        adico["intercept"] = intercept
        adico["r_value"] = r_value
        adico["p_value"] = r_value
        adico["std_err"] = std_err
        stat_df[item] = adico.values()
        if out:
            print("=========================")
            print(item)
            print("slope: %f    intercept: %f" % (slope, intercept))
            print("r-squared: %f" % r_value ** 2)
            print(p_value)
            print()
    return stat_df


corr_df = regress(latency_gain_df, out=False)

#%%
plt.close("all")

# TODO fix bug with corr_df
def plot_scatter(df, stat_df=None, num=8, shareAxes=True):
    if shareAxes:
        sharex = True
        sharey = True
    else:
        sharex = False
        sharey = False
    fig, axs = plt.subplots(
        2, num // 2, sharex=sharex, sharey=sharey, tight_layout=True, figsize=(14, 8)
    )
    axs = axs.flatten()
    #    if kind == '_x':
    #        title = 'latency advance (ms) : histogram + median value'
    #    elif kind == '_y':
    #        title = 'amplitude gain (% center test peak) : histogram + median value'
    #    else:
    #        title = ''
    title = "scatter"
    fig.suptitle(title)
    # select and sort the protocols
    protocols = [item.split("_")[0] for item in df.columns]
    protocols = list(dict.fromkeys(protocols))
    protocols = sorted([item for item in protocols if "Sector" in item]) + sorted(
        [item for item in protocols if "Full" in item]
    )
    # define the colors
    colors = {"rndiso": "b", "cpcross": "gold", "cpiso": "r", "cfiso": "g"}
    # plot
    for i, proto in enumerate(protocols):
        for name in colors:
            if name in proto:
                color = colors[name]
        label = proto
        ax = axs[i]
        ax.plot(
            df[proto + "_x"], df[proto + "_y"], "o", color=color, label=label, alpha=0.5
        )
        ax.legend()
        # regression
        x0 = stat_df[proto].intercept
        print("x0: ", "x0")
        a = stat_df[proto].slope
        print("a: ", a)
        ax.plot(df[proto + "_x"], x0 + a * df[proto + "_x"], ".-", color=color)

    # remove the frames
    for i, ax in enumerate(axs):
        if i in (0, 4):
            ax.set_ylabel("amplitude")
        if i > 3:
            ax.set_xlabel("time advance")
    if shareAxes:
        limx = [ax.get_xlim() for ax in axs]
        limy = [ax.get_ylim() for ax in axs]
        mini_x = min([list(t) for t in zip(*limx)][0])
        maxi_x = max([list(t) for t in zip(*limx)][1])
        mini_y = min([list(t) for t in zip(*limy)][0])
        maxi_y = max([list(t) for t in zip(*limy)][1])
        for ax in axs:
            ax.hlines(0, mini_x, maxi_x, alpha=0.3)
            ax.vlines(0, mini_y, maxi_y, alpha=0.3)
    else:
        for ax in axs:
            lims = ax.get_ylim()
            ax.vlines(0, lims[0], lims[1], alpha=0.3)
            lims = ax.get_xlim()
            ax.hlines(0, lims[0], lims[1], alpha=0.3)

    for ax in axs:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    #            ax.yaxis.set_visible(False)
    return fig


plot_scatter(latency_gain_df, corr_df, shareAxes=True)
# plot_scatter(df, shareAxes=False)

#%%

import statsmodels.api as sm


def compute_regress_though_sm(df):
    protocols = list(dict.fromkeys([col.split("_")[0] for col in df.columns]))
    for protoc in protocols:
        x = df[protoc + "_x"]
        x = sm.add_constant(x)
        y = df[protoc + "_y"]
        sm.OLS(y, x).fit()
        model = sm.OLS(y, x).fit()
        predictions = model.predict(x)
        print(model.summary())


compute_regress_though_sm(latency_gain_df)

#%%
def compute_regress_through_scipy(df):
    protocols = list(dict.fromkeys([col.split("_")[0] for col in df.columns]))
    resdf = pd.DataFrame(index=["slope", "intercept", "r_value", "p_value", "std_err"])
    for protoc in protocols:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[[protoc + "_x", protoc + "_y"]]
        )
        resdf[protoc] = [slope, intercept, r_value, p_value, std_err]

        print("==================")
        print(protoc)
        print("slope: %1.2f    intercept: %1.2f" % (slope, intercept))
        print("r-squared: %2.1f" % r_value ** 2)
        print("p_value : %2.2f" % p_value)
    return resdf


scipy_corr_df = compute_regress_through_scipy(latency_gain_df)

#%% not debugged the following


#%%
fig2 = plt.figure(figsize=(12, 12))
plt.plot(
    cpisosecX,
    cpisosecY,
    markersize=15,
    marker=".",
    ls="",
    color="red",
    label="data points",
)
plt.plot(cpisosecX, intercept + slope * cpisosecX, "b", label="fitted line, R^2 = 0.84")


# plt.scatter(cpisosecX, cpisosecY,s= 15, color = 'red')

plt.title("Amplitude gain function of latency advance - CP-ISO SECTOR", fontsize=20)
plt.xlabel("Latency advance (ms)", fontsize=20)
plt.ylabel("Amplitude gain (% test center peak)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

#%%
slope, intercept, r_value, p_value, std_err = stats.linregress(cpisofullX, cpisofullY)
print("slope: %f    intercept: %f" % (slope, intercept))
print("r-squared: %f" % r_value ** 2)

fig2 = plt.figure(figsize=(12, 12))
plt.plot(
    cpisofullX,
    cpisofullY,
    markersize=15,
    marker=".",
    ls="",
    color="red",
    label="data points",
)
plt.plot(
    cpisofullX, intercept + slope * cpisofullX, "b", label="fitted line, R^2 = 0.37"
)


# plt.scatter(cpisosecX, cpisosecY,s= 15, color = 'red')

plt.title("Amplitude gain function of latency advance - CP-ISO FULL", fontsize=20)
plt.xlabel("Latency advance (ms)", fontsize=20)
plt.ylabel("Amplitude gain (%)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

#%%
cpisoFull2_x = np.loadtxt(cgwdir + "cp_iso_Dlat50_full2.txt")
cpisoFull2_y = np.loadtxt(cgwdir + "cp_iso_Dres50_full2.txt")
#%%
i = cpisoFull2_x.argsort()
cpisofull2X = cpisoFull2_x[i]
cpisofull2Y = cpisoFull2_y[i]

#%%
slope, intercept, r_value, p_value, std_err = stats.linregress(cpisofull2X, cpisofull2Y)
print("slope: %f    intercept: %f" % (slope, intercept))
print("r-squared: %f" % r_value ** 2)

fig2 = plt.figure(figsize=(12, 12))
plt.plot(
    cpisofull2X,
    cpisofull2Y,
    markersize=15,
    marker=".",
    ls="",
    color="red",
    label="data points",
)
plt.plot(
    cpisofull2X, intercept + slope * cpisofull2X, "b", abel="fitted line, R^2 = 0.37"
)


# plt.scatter(cpisosecX, cpisosecY,s= 15, color = 'red')

plt.title("Amplitude gain function of latency advance - CP-ISO FULL", fontsize=20)
plt.xlabel("Latency advance (ms)", fontsize=20)
plt.ylabel("Amplitude gain (%)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

#%%
