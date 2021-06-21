#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:11:03 2021

@author: cdesbois
"""

from datetime import datetime
import os

# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import config

paths = config.build_paths()
anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()

dirname = os.path.join(paths["owncFig"], "data", "gabyToCg")

file_name = os.path.join(dirname, "4100gg3_vm_cp_cx_max.svg")
# filename = os.path.expanduser('~/4100gg3_vm_cp_cx_max.svg')

file_name = os.path.join(dirname, "test.svg")

#%% load data gaby extracted from svg


def load_gaby_data(extract=False, save=False):
    """ load the data extracter from the gaby plots """
    dirname = os.path.join(paths["owncFig"], "data", "gabyToCg")
    filename = os.path.join(dirname, 'gabydf.csv')
    if extract:
        os.chdir(os.path.join(dirname, "numGaby"))
        files = [_ for _ in os.listdir() if os.path.isfile(_)]
        files = [_ for _ in files if _.endswith("csv")]
        x = np.arange(-40, 200, 0.1)
        datadf = pd.DataFrame(index=x)
        for file in files:
            name = name = file.split(".")[0]
            df = pd.read_csv(file, sep=";", decimal=",", dtype="float", names=["x", "y"])
            df = df.sort_values(by="x")
            f = interp1d(df.x, df.y, kind="linear")
            datadf[name] = f(x)
            datadf[name] = datadf[name].rolling(11, win_type="triang", center=True).mean()
        if save:
            datadf.to_csv(filename)
            print('saved {}'.format(filename))
    else:
        datadf = pd.read_csv(filename)      
        datadf = datadf.set_index(datadf.columns[0])
    return datadf


def test_plot(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for col in df.columns:
        ax.plot(df[col], label=col)
        # ax.plot(df[col].rolling(i, win_type='triang', center=True).mean(), label=col+ 'F')
    ax.set_xlim(-40, 200)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend()
    return ax


gaby_df = load_gaby_data(extract=False, save=False)
ax = test_plot(gaby_df)

#%%
def plot_cgGabyVersion(gabydf, datadf):

    fig, axes = plt.subplots(
        figsize=(11.6, 5), nrows=1, ncols=2, sharex=True, sharey=True
    )
    axes = axes.flatten()
    used_cells = []
    # inserts
    ins = []
    for ax in axes.flat:
<<<<<<< HEAD
        ins.append(ax.inset_axes([0.75,0.5,0.25,0.5]))
    # # rect = x, y, w, h
    # zoom
    zoom_xlims = [(40, 70), (20, 50)]
    zoom_ylims = [(-2, 6)] * 2
=======
        ins.append(ax.inset_axes([0.7, 0.5, 0.3, 0.5]))
    # # rect = x, y, w, h
    # zoom locations
    zoom_xlims = [(40, 70), (20, 50)]
    zoom_ylims = [(-2, 7)] * 2
>>>>>>> work
    for i, ax in enumerate(axes):
        xy = zoom_xlims[i][0], zoom_ylims[i][0]
        w = zoom_xlims[i][1] - zoom_xlims[i][0]
        h = zoom_ylims[i][1] - zoom_ylims[i][0]
        ax.add_patch(
            Rectangle(xy, w, h, fill=False, edgecolor="tab:grey", lw=2, ls=":")
        )
    # data
    df0 = gabydf.copy()
    df1 = datadf.copy()
    middle = (df1.index.max() - df1.index.min()) / 2
    df1.index = (df1.index - middle) / 10
    cells = list(set([_.split("_")[0] for _ in df1.columns]))
    # limit the date time range
    # df = df.loc[-200:200]
    df1 = df1.loc[-42.5:206]
    dfs = [df0, df1]

    colors = [config.std_colors()[_] for _ in "red red yellow yellow k".split()]
    style = ["-", ":", "-", ":", "-"]
    linewidth = [2, 3, 2, 3, 2]
    alpha = [0.8, 0.8, 1, 1, 0.7]

    # plot
    for i, ax in enumerate(axes):
        insert = ins[i]
        df = dfs[i]
        if i == 0:
            cols = [
                "4100gg3_sc",
                "4100gg3_s0",
                "4100gg3_x_sc",
                "4100gg3_x_s0",
                "4100gg3_0c",
            ]
            cell = cols[0].split("_")[0]
        else:
            cell = cells[0]
            cell = "1516gcxg2"
            cols = [_ for _ in df1.columns if cell in _]
        # if anot:
        #     ax.set_title(cell, color="tab:grey")
        used_cells.append(cell)
        for j, col in enumerate(cols):
            ax.plot(
                df[col],
                linestyle=style[j],
                linewidth=linewidth[j],
                color=colors[j],
                alpha=alpha[j],
                # label=labels[j]
            )
            # insert plotting
            if col.endswith("0") or "_so" in col:
                # don't plot 'no center'
                continue
            insert.plot(
                df[col],
                linestyle=style[j],
                linewidth=linewidth[j],
                color=colors[j],
                alpha=alpha[j],
                # label=labels[j],
            )
    # to adapt to gaby
    for ax in axes:
        ax.set_xlim(-40, 200)
        ax.set_ylim(-5, 15)
        ax.set_yticks(range(0, 14, 2))
    for i, ax in enumerate(ins):
        ax.set_xlim(zoom_xlims[i])
        ax.set_ylim(zoom_ylims[i])
        ax.axhline(y=0, alpha=0.5, color="k")
        ax.set_yticks([0, 3, 6])
    for i, ax in enumerate(fig.get_axes()):
        ax.axhline(y=0, alpha=0.5, color="k")
        ax.axvline(x=0, alpha=0.5, color="k")
        ax.set_xlabel("Time (ms)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.set_ylabel("Membrane Potential (mV)")
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "GabyToCg:plot_cgGabyVersion",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        txt = '{}   |   {}'.format(used_cells[0], used_cells[1])
        fig.text(0.5, 0.01, txt, ha="center", va="bottom", alpha=0.4)        
    return fig


plt.close("all")
anot = True

file = "cg_specificity.xlsx"
file_name = os.path.join(dirname, "sources", file)
data_df = pd.read_excel(file_name)

fig = plot_cgGabyVersion(gaby_df, data_df)

save = False
if save:
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "gabyToCg")
    for ext in [".svg", ".png", ".pdf"]:
        file_name = os.path.join(dirname, "cgGabyVersion" + ext)
        fig.savefig(file_name)
    name = "f5_transition_bottom"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(os.path.join(paths["save"], (name + ext)))
