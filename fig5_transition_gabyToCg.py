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


def load_examples(display=False):
    file = "example_cardVsRadial.hdf"
    data_loadname = os.path.join(paths["hdf"], file)
    gabyexampledf = pd.read_hdf(data_loadname, key="card")
    print("loaded {} ({}) ".format(file, "card"))
    cgexampledf = pd.read_hdf(data_loadname, key="rad")
    print("loaded {} ({}) ".format(file, "rad"))
    if display:
        for key, df in zip(
            ["cardinal_data", "radial_data"], [gabyexampledf, cgexampledf]
        ):
            print("=" * 20, "{}".format(key))
            for col in df.columns:
                print(col)
            print()
    return gabyexampledf, cgexampledf


def plot_cgGabyVersion(gabydf, cgdf):
    """
    two cells examples to explain the change in protocols
    """

    # # to build
    # gabydf = gaby_example_df
    # cgdf = cg_example_df

    fig, axes = plt.subplots(
        figsize=(8.6, 4), nrows=1, ncols=2, sharex=True, sharey=True
    )
    axes = axes.flatten()
    used_cells = []
    # inserts
    ins = []
    for ax in axes.flat:
        ins.append(ax.inset_axes([0.7, 0.6, 0.3, 0.4]))
    # rect = x, y, w, h
    # zoom locations
    zoom_xlims = [(40, 70), (20, 50)]
    zoom_ylims = [(-2, 7)] * 2
    for i, ax in enumerate(axes):
        xy = zoom_xlims[i][0], zoom_ylims[i][0]
        w = zoom_xlims[i][1] - zoom_xlims[i][0]
        h = zoom_ylims[i][1] - zoom_ylims[i][0]
        ax.add_patch(
            Rectangle(xy, w, h, fill=False, edgecolor="tab:grey", lw=2, ls=":")
        )
    # data
    df0 = gabydf.loc[-40:199].copy()
    df1 = cgdf.loc[-40:199].copy()
    cells = list({_.split("_")[0] for _ in df1.columns})
    # limit the date time range
    dfs = [df0, df1]

    colors = [config.std_colors()[_] for _ in "red red yellow yellow k".split()]
    style = ["-", ":", "-", ":", "-"]
    linewidth = [2, 2, 1.5, 2, 1.5]
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
        ax.set_xlabel("Time (ms)")
    for i, ax in enumerate(ins):
        ax.set_xlim(zoom_xlims[i])
        ax.set_ylim(zoom_ylims[i])
        ax.axhline(y=0, alpha=0.5, color="k")
        ax.set_yticks([0, 3, 6])
    for i, ax in enumerate(fig.get_axes()):
        ax.axhline(y=0, alpha=0.5, color="k")
        ax.axvline(x=0, alpha=0.5, color="k")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if i in [0, 2]:
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
        txt = "{}   |   {}".format(used_cells[0], used_cells[1])
        fig.text(0.5, 0.01, txt, ha="center", va="bottom", alpha=0.4)
    return fig


plt.close("all")

if "gaby_example_df" not in dir() or "cg_example_df" not in dir():
    gaby_example_df, cg_example_df = load_examples(display=False)

fig = plot_cgGabyVersion(gaby_example_df, cg_example_df)

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

#%% save data
