#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 28 16:55:32 2021

@author: cdesbois
"""

import os
from datetime import datetime
from importlib import reload
from bisect import bisect

# from itertools import zip_longest

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import config

std_colors = config.std_colors()
speed_colors = config.speed_colors()

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

paths["data"] = os.path.join(paths.get("owncFig"), "data")

# NB all limits are lower ones!


def load_speed_data():
    """
    load the data files

    Returns
    -------
    bddf : pd.DataFrame
        baudot data
    cgdf : pd.DataFrame
        centrigabor data.

    """
    name = "baudot.csv"
    filename = os.path.join(paths.get("data"), name)
    bddf = pd.read_csv(filename)
    bddf.columns = [_.strip() for _ in bddf.columns]
    # to have the lower interval limit
    bddf.optiMax = bddf.optiMax - 25

    name = "neuron_props_speed.xlsx"
    filename = os.path.join(paths.get("data"), "averageTraces", name)
    cgdf = pd.read_excel(filename)
    cols = [st.strip() for st in cgdf.columns]
    cols = [st.lower() for st in cols]
    cols = [st.replace("(", "_") for st in cols]
    cols = [st.replace(")", "") for st in cols]
    cgdf.columns = cols
    return bddf, cgdf


def build_summary(bddf: pd.DataFrame, speeddf: pd.DataFrame) -> pd.DataFrame:
    """
    class the number of cells

    Parameters
    ----------
    bddf : pd.DataFrame
        baudot data.
    cgdf : pd.DataFrame
        centrigabor data.

    Returns
    -------
    pd.DataFrame
        summary dataframe (nb of cells, upper interval spped limit).

    """
    # count number of cells
    def class_speed(speed):
        "return lower limit in class"
        lims = range(100, 500, 25)
        i = bisect(lims, speed)
        return lims[i - 1]

    speeddf["optiMax"] = speeddf.speed_isi0.apply(lambda x: class_speed(x))
    speednum = dict(speeddf.optiMax.value_counts())
    bdnum = dict(bddf.set_index("optiMax"))

    summarydf = pd.DataFrame(index=range(50, 525, 25))
    summarydf = summarydf.join(pd.DataFrame(bdnum))
    summarydf.columns = ["bd_cells"]
    summarydf["cg_cells"] = pd.Series(speednum)
    summarydf = summarydf.fillna(0)
    summarydf = summarydf.astype(int)
    return summarydf


def load_cgpopdf() -> pd.DataFrame:
    """
    load the whole centrigabor population  (n=37)

    Returns
    -------
    pandas dataframe
    """
    name = "centrigabor_pop_db.xlsx"

    filename = os.path.join(paths.get("data"), "averageTraces", name)
    popdf = pd.read_excel(filename)
    popdf.columns = [col.lower().strip() for col in popdf.columns]
    return popdf


def load_bringuier() -> pd.DataFrame:
    """
    load the bringuier data

    Returns
    -------
    brdf : pd.DataFrame
    """
    filename = os.path.join(paths.get("data"), "bringuier.csv")
    brdf = pd.read_csv(filename, sep="\t", decimal=",")
    brdf = pd.DataFrame(brdf)
    # brdf.speed_upper = brdf.speed_upper  ## * 1000  # 1mm / visual°
    # unstack
    temp = brdf.impulse.fillna(0) - brdf.long_bar.fillna(0)
    temp = temp.apply(lambda x: x if x >= 0 else 0)
    brdf.impulse = temp
    brdf = brdf.fillna(0)
    brdf = brdf.rename(columns={"speed_upper": "speed_lower"})
    return brdf


def load_gmercier() -> pd.DataFrame:
    """
    load the gerardmercier (hexagabor) data

    Returns
    -------
    df : pd.DataFrame
    """
    name = "gmercier.csv"
    filename = os.path.join(paths.get("data"), name)
    gmdf = pd.read_csv(filename, sep="\t")
    del gmdf["speed_high"]
    return gmdf


def load_gmercier2() -> pd.DataFrame:
    """
    load the gerardmercier (hexagabor) data

    Returns
    -------
    df : pd.DataFrame
    """
    name = "gmercier2.csv"
    filename = os.path.join(paths.get("data"), name)
    gmdf2 = pd.read_csv(filename, sep="\t", decimal=",")
    gmdf2 = gmdf2.set_index("cell")
    # remove empty lines
    gmdf2 = gmdf2.dropna(how="all")
    return gmdf2


# speed and baudot
bd_df, speed_df = load_speed_data()
summary_df = build_summary(bd_df, speed_df)
# replaced cgdf by spdf (speeddf)

# gmercier
gm_df = load_gmercier()
gm_df2 = load_gmercier2()

# binguier
br_df = load_bringuier()

# cg population
pop_df = load_cgpopdf()


def samebin(
    popdf: pd.DataFrame = pop_df,
    spdf: pd.DataFrame = speed_df,
    bddf: pd.DataFrame = bd_df,
    gmdf: pd.DataFrame = gm_df,
    brdf: pd.DataFrame = br_df,
) -> pd.DataFrame:
    """
    resample and group the data to have all using the same bin

    Parameters
    ----------
    popdf : pd.DataFrame <-> centrigabor population data
    spdf : pd.DataFrame <-> centrigabor speed data
    bddf : pd.DataFrame <-> baudot data
    gmdf : pd.DataFrame <-> gérard mercier data
    brdf : pd.DataFrame <-> bringuier data

    Returns
    -------
    pd.DataFrame <-> data with the same bin

    """

    def compute_speed_histo(speeds: pd.Series, bins: int = 40):
        # speeds = df.speed/1000
        heights, xvals = np.histogram(speeds, bins=bins, range=(0, 1))
        # normalize
        # height = height/np.sum(height)
        width = (max(xvals) - min(xvals)) / (len(xvals) - 1)
        return xvals[:-1], heights, width

    # res dataframe 20 bins 0 to 1 range
    bineddf = pd.DataFrame(index=[_ / 20 for _ in range(0, 20, 1)])
    # centrigabor population
    _, heights_cgpop, _ = compute_speed_histo(popdf.speed / 1000, bins=20)
    bineddf["cgpop"] = heights_cgpop
    # speed centrigabor pop
    _, heights_cgspeed, _ = compute_speed_histo(spdf.speed_isi0 / 1000, bins=20)
    bineddf["cgspeed"] = heights_cgspeed
    # baudot -> resample to double bin width
    bdfcop = bddf.copy()
    bdfcop.loc[-1] = [150, 0]  # append row to match the future scale
    bdfcop.index = bdfcop.index + 1
    bdfcop = bdfcop.sort_index()
    bdfcop = bdfcop.set_index("optiMax")
    bdfcop.index = bdfcop.index / 1000
    bdfcop["cells"] = bdfcop.nbCell.shift(-1).fillna(0)
    bdfcop.cells += bdfcop.nbCell
    bdfcop = bdfcop.drop(bdfcop.index[1::2])
    bdfcop = bdfcop.drop(columns=["nbCell"])
    bineddf["bd"] = bdfcop
    # gmercier
    temp = gmdf.set_index("speed_low")
    temp.index = temp.index / 1000
    bineddf["gm"] = temp
    # bringuier
    temp = brdf.set_index("speed_lower")
    # remove upper limit
    temp = temp.drop(index=[100])
    # rename columns
    cols = ["br_" + st for st in temp.columns]
    bineddf[cols[0]] = temp[temp.columns[0]]
    bineddf[cols[1]] = temp[temp.columns[1]]
    # fill and convert to integer
    bineddf = bineddf.fillna(0).astype("int")
    return bineddf


bined_df = samebin()

# TEST mean bins values
# sum(df.impulse * meanBinValue) / sum(df.impulse)

# impulse = ((speed_lower + 0.025) * impulse)
# brdf['impulse_contr'] = (brdf.speed_lower + 0.025) * brdf['impulse']
# brdf['long_bar_contr'] = (brdf.speed_lower + 0.025) * brdf['long_bar']
# np.average(values=brdf.speed_lower + 0.025, weights=brdf.impulse)
# np.average(brdf.speed_lower + 0.025, weights=brdf.impulse)


def weighted_avg_and_std(values: pd.Series, weights: pd.Series):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


w_mean, w_std = weighted_avg_and_std(br_df.speed_lower + 0.025, br_df.impulse)
print(f"impulse : {w_mean:.2f} ± {w_std:.2f} (weighted mean ± std) ")
w_mean, w_std = weighted_avg_and_std(br_df.speed_lower + 0.025, br_df.long_bar)
print(f"long_bar : {w_mean:.2f} ± {w_std:.2f} (weighted mean ± std)")

#%%


def stats_brdf(brdf: pd.DataFrame):
    """perfoem a weighted stat from histogram data"""
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    brdfcop = brdf.copy()
    brdfcop = brdfcop.drop(brdfcop.index[-1], axis=0)
    brdfcop.speed_lower += 0.025
    brdfcop.impulse = brdfcop.impulse.astype(int)

    for col in brdfcop.columns[1:]:
        expansion = []
        for item in brdfcop.iterrows():
            temp = [item[1].speed_lower] * int(item[1][col])
            if len(temp) > 0:
                expansion.extend(temp)
        expansion = pd.Series(expansion)
        res = expansion.aggregate(["mean", "std", "median", "mad"])
        print(f"{col:-^20}")
        print(res, "\n")


# stats_brdf(br_df)


def stats_bineddf(bineddf=bined_df):
    """
        compute the statistical description related to the the histogrammes

        Parameters
        ----------
        bineddf : pd.DataFrame, optional (default is bined_df)

        Returns
    cd     -------
        None.

    """
    bineddf = bineddf.copy()
    bineddf["sum_down"] = bineddf["br_long_bar"] + bineddf["br_impulse"] + bineddf["gm"]
    bineddf["sum_up"] = bineddf["cgpop"] + bineddf["bd"]
    downs = ["br_long_bar", "br_impulse", "gm"]
    ups = ["cgpop", "bd"]

    for loc in ups + downs + ["sum_up", "sum_down"]:
        pop = []
        for i, k in bineddf[loc].iteritems():
            for _ in range(k):
                pop.append(i + 0.025)
        ser = pd.Series(pop)
        print(f"{'-'*5} {loc} {'-'*5}")
        print(f"mode={ser.mode()[0]}")
        print(ser.agg(["count", "mean", "std", "median", "mad"]))
        print()


stats_bineddf()


#%%


def plot_optimal_speed(bddf: pd.DataFrame = bd_df, popdf: pd.DataFrame = pop_df):
    """plot speed"""
    # prepare data
    height_cg, x = np.histogram(popdf.speed, bins=18, range=(50, 500))
    x = x[:-1]
    toplotdf = pd.DataFrame(index=range(50, 525, 25))
    toplotdf["popcg"] = pd.Series(data=height_cg, index=x)
    toplotdf["bd"] = bddf.set_index("optiMax")
    toplotdf = toplotdf.fillna(0)
    align = "edge"  # ie right edge
    width = (toplotdf.index.max() - toplotdf.index.min()) / (len(toplotdf) - 1)

    # plot
    fig = plt.figure(figsize=(11.6, 5))
    ax = fig.add_subplot(111)
    # NB ax.bar, x value = lower
    ax.bar(
        toplotdf.index,
        height=toplotdf.popcg,
        width=width,
        align=align,
        color="w",
        edgecolor="k",
        alpha=0.6,
        label="cengrigabor",
    )
    ax.bar(
        toplotdf.index,
        height=toplotdf.bd,
        bottom=toplotdf.popcg,
        width=width,
        align=align,
        color="tab:gray",
        edgecolor="k",
        alpha=0.6,
        label="baudot",
    )
    txt = f"n= {toplotdf.popcg.sum():.0f} cells"
    ax.text(
        x=0.8, y=0.6, s=txt, color="k", va="bottom", ha="left", transform=ax.transAxes
    )
    txt = f"n= {toplotdf.bd.sum():.0f} cells"
    ax.text(
        x=0.8,
        y=0.5,
        s=txt,
        color="tab:gray",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )
    ax.set_xlabel("optimal apparent speed ".title() + "(°/sec)")
    ax.set_ylabel("nb of Cells")
    ax.legend()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "speed_baudot.py:plot_optimal_speed",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


plt.close("all")
anot = True

figure = plot_optimal_speed(bd_df)
save = False
if save:
    file = "optSpeed.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%%
plt.close("all")


def plot_optimal_bringuier(gdf: pd.DataFrame = bined_df) -> plt.Figure:
    """
    plot an histogram of the optimal cortical horizontal speed
    Parameters
    ----------
    df = pandas dataFrame
    Returns
    -------
    fig = matplotlib.pyplot.figure

    """
    fig, ax = plt.subplots(figsize=(4.3, 4), nrows=1, ncols=1)
    # gerenal features
    x = gdf.index
    width = (max(x) - min(x)) / (len(x) - 1) * 0.98
    align = "edge"

    txt = f"Bar n={gdf.br_long_bar.sum():.0f}"
    ax.bar(
        x,
        height=gdf.br_long_bar,
        width=width,
        align=align,
        alpha=0.8,
        color=std_colors["blue"],
        edgecolor="k",
        label=txt,
    )
    gdf["pool"] = gdf.br_long_bar
    txt = f"SN n={gdf.br_impulse.sum():.0f}"
    ax.bar(
        x,
        height=gdf.br_impulse,
        bottom=gdf.pool,
        width=width,
        align=align,
        color=std_colors["dark_green"],
        edgecolor="k",
        alpha=0.8,
        label=txt,
    )
    gdf.pool += gdf.br_impulse
    txt = f"Gabor n={gdf.gm.sum():.0f}"
    ax.bar(
        x,
        gdf.gm,
        bottom=gdf.pool,
        width=width,
        align=align,
        color=std_colors["green"],
        alpha=0.7,
        edgecolor="k",
        label=txt,
    )
    # txt = 'Apparent Speed of Horizontal Propagation (ASHP) m/s'
    txt = "Propagation Speed (mm/ms)"
    ax.set_xlabel(txt)
    ax.set_ylabel("Nb of measures")
    ax.legend()

    for ax in fig.get_axes():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    lims = ax.get_xlim()
    ax.set_xlim(0, lims[1])
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "speed_baudot.py:plot_optimal_bringuier",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


plt.close("all")
figure = plot_optimal_bringuier(bined_df)

save = False
if save:
    file = "optSpreedBringuier.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)
    # update current
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "current", "fig")
    file = "o1_optSpeedBringuier"
    for ext in [".png", ".pdf", ".svg"]:
        file_name = os.path.join(dirname, (file + ext))
        figure.savefig(file_name)

#%%
def plot_both(bineddf: pd.DataFrame = bined_df):
    """
    plot both histo (up and down)

    Parameters
    ----------
    bineddf : pd.DataFrame, bined_df.

    Returns
    -------
    fig : plt.figure

    """

    fig, ax = plt.subplots(figsize=(4.3, 16))
    # gerenal features
    x = bineddf.index
    width = (max(x) - min(x)) / (len(x) - 1) * 0.98
    align = "edge"

    bineddf["minuspool"] = 0  # ref
    # plot mirror low
    ax.bar(
        x,
        height=(bineddf.br_long_bar * -1),
        width=width,
        align=align,
        alpha=0.3,
        color=std_colors["blue"],
        edgecolor="tab:grey",
    )
    bineddf["minuspool"] += bineddf.br_long_bar * -1  # ref
    ax.bar(
        x,
        height=(bineddf.br_impulse * -1),
        bottom=bineddf.minuspool,
        # bottom=(bineddf.br_long_bar * -1),
        width=width,
        align=align,
        alpha=0.3,
        color=std_colors["dark_green"],
        edgecolor="tab:grey",
    )
    bineddf["minuspool"] += bineddf.br_impulse * -1  # ref
    txt = f"Gabor n={bineddf.gm.sum():.0f}"
    ax.bar(
        x,
        height=(bineddf.gm * -1),
        bottom=bineddf.minuspool,
        width=width,
        align=align,
        color=std_colors["green"],
        alpha=0.3,
        edgecolor="tab:grey",
        # label=txt,
    )
    bineddf.pool += bineddf.gm * -1

    bineddf["pool"] = 0  # ref
    txt = f"Radial n={bineddf.cgpop.sum():.0f}"
    ax.bar(
        x,
        bineddf.cgpop,
        bottom=bineddf.pool,
        width=width,
        align=align,
        color=speed_colors["red"],
        alpha=0.6,
        edgecolor="k",
        label=txt,
    )
    bineddf.pool += bineddf.cgpop
    txt = f"Cardinal n={bineddf.bd.sum():.0f}"
    ax.bar(
        x,
        bineddf.bd,
        bottom=bineddf.pool,
        width=width,
        align=align,
        color=speed_colors["yellow"],
        alpha=0.6,
        edgecolor="k",
        label=txt,
    )
    bineddf.pool += bineddf.bd
    # txt = f"Gabor n={bineddf.gm.sum():.0f}"
    # ax.bar(
    #     x,
    #     bineddf.gm,
    #     bottom=bineddf.pool,
    #     width=width,
    #     align=align,
    #     color=speed_colors["orange"],
    #     alpha=0.6,
    #     edgecolor="k",
    #     label=txt,
    # )
    # bineddf.pool += bineddf.gm

    txt = "Inferred Cortical Speed (mm/ms)"
    ax.set_xlabel(txt)
    ax.set_ylabel("Nb of cells")
    ax.legend()
    # set the number off (all) cells in positive
    ax.yaxis.set_major_locator(mticker.MaxNLocator(7))
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc[1:-1]))
    ax.set_yticklabels([abs(int(_)) for _ in ticks_loc[1:-1]])

    # # bottom
    # ax = axes[1]
    # txt = 'Bar n={:.0f}'.format(bineddf.br_long_bar.sum())
    # ax.bar(x, height=bineddf.br_long_bar, width=width, align=align, alpha=0.8,
    #        color=std_colors['blue'], edgecolor='k',
    #        label=txt)
    # txt = 'SN n={:.0f}'.format(bineddf.br_impulse.sum())
    # ax.bar(x, height=bineddf.br_impulse, bottom=bineddf.br_long_bar, width=width,
    #        align=align, color=std_colors['green'],
    #        edgecolor='k', alpha=0.8, label=txt)
    # # txt = 'Apparent Speed of Horizontal Propagation (ASHP) m/s'
    # txt = 'Propagation Speed (mm/ms)'
    # ax.set_xlabel(txt)
    # ax.set_ylabel('Nb of measures')
    # ax.legend()

    for ax in fig.get_axes():
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    lims = ax.get_xlim()
    ax.set_xlim(0, lims[1])
    ax.set_xlim(0, 0.6)

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99, 0.01, "speed_baudot.py:plot_both", ha="right", va="bottom", alpha=0.4
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        # fig.text(0.5, 0.01, 'cortical speed',
        #          ha='center', va='bottom', alpha=0.4)
    return fig


def save_fig10_data_histo(bineddf: pd.DataFrame = bined_df, do_save: bool = False):
    """save the data used to build the figure"""

    data_savename = os.path.join(paths["figdata"], "fig10.hdf")
    # histogram
    key = "histo"
    copydf = bineddf.copy()
    copydf = copydf.drop(axis=1, columns=["cgspeed", "pool"])
    cols = copydf.columns
    cols = [_.replace("cgpop", "radial") for _ in cols]
    cols = [_.replace("bd", "cardinal") for _ in cols]
    cols = [_.replace("gm", "twoStrokes") for _ in cols]
    cols = [_.replace("br", "bringuier") for _ in cols]
    copydf.columns = cols
    print("-" * 20, f"{os.path.basename(data_savename)}({key})")
    for item in cols:
        print(item)
    print()
    if do_save:
        copydf.to_hdf(data_savename, "histo")


plt.close("all")
figure = plot_both(bined_df)
save = False
if save:
    file = "optSpreedBoth.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)
    # update current
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "current", "fig")
    file = "f10_optSpreedBoth"
    for ext in [".png", ".pdf", ".svg"]:
        file_name = os.path.join(dirname, (file + ext))
        figure.savefig(file_name)

save_fig10_data_histo(bined_df, False)

#%%
# targetbins = list(np.linspace(0, 1, 21))
# targetbins = [round(_, 2) for _ in targetbins]


def hist_summary(
    brdf: pd.DataFrame = br_df,
    summarydf: pd.DataFrame = summary_df,
    speeddf: pd.DataFrame = speed_df,
    popdf: pd.DataFrame = pop_df,
    gmdf: pd.DataFrame = gm_df,
    maxx=1,
) -> plt.Figure:
    """
    histogramme distribution of the main results from the lab (is one hist by experiment)

    Parameters
    ----------
    brdf : pd.DataFrame <-> bringuier
    summarydf : pd.DataFrame <-> summary
    cgdf : pd.DataFrame <-> centrigabor
    df : pd.DataFrame
    gmdf : pd.DataFrame <-> gerard mercier
    maxx : float (default is 1) <-> max speed to display (xscale)

    >>>> call = br_df, summary_df, speed_df, pop_df, gm_df

    Returns
    -------
    plt.Figure
    """

    def compute_speed_histo(speeds, bins=40):
        # speeds = df.speed/1000
        height, x = np.histogram(speeds, bins=bins, range=(0, 1))
        # normalize
        height = height / np.sum(height)
        width = (max(x) - min(x)) / (len(x) - 1)
        return x[:-1], height, width

    fig, axes = plt.subplots(
        figsize=(8.6, 12), nrows=2, ncols=3, sharey=True, sharex=True
    )
    axes = axes.flatten()

    # bringuier flash
    ax = axes[0]
    x = brdf.speed_lower.tolist()
    x = [round(_, 2) for _ in x]
    height_bar = (brdf.impulse) / brdf.impulse.sum().tolist()
    align = "edge"
    x[-1] = 1.05  # for continuous range
    width = max(x) / (len(x) - 1)
    ax.bar(
        x,
        height=height_bar,
        width=width,
        align=align,
        alpha=1,
        color="tab:green",
        edgecolor="k",
        label="impulse bringuier",
    )
    txt = f"n = 37 \n ({int(brdf.impulse.sum())} measures)"
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
    moy = (brdf.impulse * (brdf.speed_lower + 0.025))[:-1].sum() / brdf.impulse.sum()
    txt = f"mean ~ {moy:.2f}"
    ax.text(
        x=0.7,
        y=0.6,
        s=txt,
        va="top",
        ha="center",
        color="tab:green",
        transform=ax.transAxes,
    )
    ax.axvline(moy, color="tab:green")
    ax.legend()

    # bringuier bar
    ax = axes[1]
    x = brdf.speed_lower.tolist()
    x = [round(_, 2) for _ in x]
    height_bar = (brdf.long_bar) / brdf.long_bar.sum().tolist()
    align = "edge"
    x[-1] = 1.05  # for continuous range
    width = max(x) / (len(x) - 1)
    ax.bar(
        x,
        height=height_bar,
        width=width,
        align=align,
        alpha=1,
        color="tab:blue",
        edgecolor="k",
        label="bar bringuier",
    )
    txt = f"n = 27 \n ({int(brdf.long_bar.sum())} measures)"
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
    moy = (brdf.long_bar * (brdf.speed_lower + 0.025)[:-1]).sum() / brdf.long_bar.sum()
    txt = f"mean ~ {moy:.2f}"
    ax.text(
        x=0.7,
        y=0.6,
        s=txt,
        va="top",
        ha="center",
        color="tab:blue",
        transform=ax.transAxes,
    )
    ax.axvline(moy, color="tab:blue")
    ax.legend()

    # baudot
    ax = axes[2]
    x = summarydf.index / 1000
    height = summarydf.bd_cells / summarydf.bd_cells.sum()
    width = (max(x) - min(x)) / (len(x) - 1)
    # width = 0.02
    ax.bar(
        x,
        height,
        width=width,
        color="tab:purple",
        edgecolor="k",
        alpha=0.8,
        label="baudot",
    )
    txt = f"n= {int(summarydf.bd_cells.sum())}"
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
    moy = (summarydf.bd_cells * x).sum() / summarydf.bd_cells.sum()
    txt = f"mean~ {moy:.2f}"
    ax.text(
        x=0.7,
        y=0.6,
        s=txt,
        va="top",
        ha="center",
        color="tab:purple",
        transform=ax.transAxes,
    )
    ax.axvline(moy, color="tab:purple")
    ax.legend()

    # gerard mercier
    ax = axes[3]
    x = gmdf.speed_low / 1000
    height = gmdf.cells / gmdf.cells.sum()
    width = (max(x) - min(x)) / (len(x) - 1)
    ax.bar(
        x,
        height,
        width=width,
        color="tab:brown",
        edgecolor="k",
        align="edge",
        alpha=0.8,
        label="gmercier",
    )
    txt = f"n= {gmdf.cells.sum()}"
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
    moy = (gmdf.cells * x).sum() / gmdf.cells.sum()
    txt = f"mean ~ {moy:.2f}"
    ax.text(
        x=0.7,
        y=0.6,
        s=txt,
        va="top",
        ha="center",
        color="tab:brown",
        transform=ax.transAxes,
    )
    ax.axvline(moy, color="tab:brown")
    ax.legend()

    # centripop
    ax = axes[4]
    ax.bar(
        *compute_speed_histo(popdf.speed / 1000),
        color="tab:red",
        edgecolor="k",
        alpha=0.8,
        label="cg population",
    )
    txt = f"n= {len(popdf)}"
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)

    txt = f"mean ± std : \n {(popdf.speed / 1000).mean():.2f} \
        ± {(popdf.speed / 1000).std():.2f}"
    ax.text(
        x=0.7,
        y=0.7,
        s=txt,
        va="top",
        ha="center",
        color="tab:red",
        transform=ax.transAxes,
    )
    ax.axvline((popdf.speed / 1000).mean(), color="tab:red")
    ax.legend()

    # speed pop
    ax = axes[5]
    ax.bar(
        *compute_speed_histo(speeddf.speed_isi0 / 1000),
        color="tab:orange",
        edgecolor="k",
        alpha=0.8,
        label="speed population",
    )
    txt = "n = {}".format(len(speeddf))
    ax.text(x=0.6, y=0.8, s=txt, va="top", ha="left", transform=ax.transAxes)
    txt = "mean ± std : \n {(speeddf.speed_isi0 / 1000).mean():.2f} \
        ± {(speeddf.speed_isi0 / 1000).std():.2f}"
    ax.text(
        x=0.7,
        y=0.7,
        s=txt,
        va="top",
        ha="center",
        color="tab:orange",
        transform=ax.transAxes,
    )
    ax.axvline((speeddf.speed_isi0 / 1000).mean(), color="tab:orange")
    ax.legend()

    fig.suptitle("summary for speed")
    ax.set_xlim(0, maxx)
    for ax in fig.get_axes():
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
    for ax in axes[2:]:
        ax.set_xlabel("speed (m/s)")

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "speed_baudot.py:hist_summary",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)

    fig.tight_layout()
    return fig


plt.close("all")

fig1 = hist_summary()
fig2 = hist_summary(maxx=0.5)
save = False
if save:
    file = "hist_summary.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    fig1.savefig(file_name)
    file = "hist_summary05.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    fig2.savefig(file_name)


#%% summary dot plots

plt.rcParams["axes.xmargin"] = 0.05
plt.rcParams["axes.ymargin"] = 0.05
plt.close("all")


def dotplot_latency(bineddf: pd.DataFrame = bined_df) -> plt.Figure:
    """
    dot plot of the latencies for all experiments

    Parameters
    ----------
    bineddf : pd.DataFrame
        the equaly binned data

    Returns
    -------
    fig : TYPE
        dot_plot.

    """
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111)
    # y = df.index.tolist()
    y = (bined_df.index + 0.025).tolist()
    populations = ["br_long_bar", "br_impulse", "bd", "gm", "cgpop", "cgspeed"]
    popsig = [
        "bringuier_bars",
        "bringuier_impulses",
        "baudot",
        "gerardMercier",
        "centrigabor_pop",
        "centrigabor_speedPop",
    ]
    labels = dict(zip(populations, popsig))
    colors = [
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:brown",
        "tab:red",
        "tab:orange",
    ]
    for i, pop in enumerate(populations):
        x = (bineddf[pop] / bineddf[pop].sum()).tolist()
        x = list(map(lambda x: x if x > 0 else np.nan, x))
        ax.plot(
            x,
            y,
            marker="o",
            ls="",
            markeredgecolor="w",
            markerfacecolor=colors[i],
            alpha=0.6,
            markeredgewidth=1.5,
            markersize=16,
            label=labels[pop],
        )
    ax.legend()
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    fig.suptitle("latency (proportion of cells)")
    ax.set_ylabel("optimal speed (m/sec)")
    # ax.set_xlabel('proportion of cells')´
    ticks = np.linspace(0, 1, 11)
    ax.set_yticks(ticks)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "speed_baudot.py:dotplot_latency",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


figure = dotplot_latency(bined_df)
save = False
if save:
    file = "dotplot_latency.pdf"
    dirname = os.path.join(paths.get("owncFig"), "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%%
# get an estimate of the distribution
bined_df["sum_up"] = bined_df["br_long_bar"] + bined_df["br_impulse"] + bined_df["gm"]
bined_df["est_mean"] = bined_df.index
bined_df.est_mean += 0.025
bined_df.est_mean *= bined_df.sum_up
(bined_df.est_mean / bined_df.sum_up).describe()
