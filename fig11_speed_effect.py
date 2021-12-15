#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:49:28 2021

@author: cdesbois
"""

import os
from datetime import datetime

from importlib import reload
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import fig_proposal as figp
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

paths = config.build_paths()
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "fillinIn", "indFill_popFill"
)

#%%


def load_pop_datafile(key="fillsig", display=True):
    """load the popfilldf dataframe (for fig 6 & 9)"""
    try:
        key in ["pop", "pop2sig", "pop3sig", "fillsig", "speed"]
    except NameError:
        print("key shoud be in ['pop', 'pop2sig', 'pop3sig', 'fillsig', 'speed']")
        return pd.DataFrame()
    loadfile = "populations_traces.hdf"
    loaddirname = paths["hdf"]
    loadfilename = os.path.join(loaddirname, loadfile)
    df = pd.read_hdf(loadfilename, key)
    print("-" * 20)
    print("loaded {} population".format(key))
    if display:
        print("=" * 20, "{}({})".format(loadfile, key))
        for column in sorted(df.columns):
            print(column)
        print()
    return df


def plot_speed(popspeeddf, substract=False, spread=[0, 1], anot=True):
    """ex fig 4"""

    # to build
    popspeeddf = popspeed_df.copy()
    popspeeddf.columns = [_.replace("sect", "s") for _ in popspeeddf.columns]
    traces = [_ for _ in popspeeddf.columns if "_se" not in _]
    colors = [
        config.speed_colors()[_] for _ in "k red dark_orange orange yellow".split()
    ]
    alphas = [0.8, 1, 0.8, 0.8, 1]
    linewidths = 1.5
    alphafill = 0.3

    df = popspeeddf.copy()
    #    cols = ["centerOnly", "100%", "70%", "50%", "30%"]

    if substract:
        ref = df[traces[0]]
        df = df.subtract(ref, axis=0)
        # stack
        # stacks = []
        # for i, col in enumerate(df.columns[:-5:-1]):
        #     df[col] += i / 10 * 2
        #     stack.append(i / 10 * 2)
    size = (8, 5)
    fig = plt.figure(figsize=size)
    # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    se = False  # increase linewith only if one se is diplayed
    for i, trace in enumerate(traces):
        ax.plot(
            df[trace],
            color=colors[i],
            alpha=alphas[i],
            label=trace,
            linewidth=linewidths,
        )
        if i in spread:
            se = True
            ax.fill_between(
                df.index,
                df[trace + "_seup"],
                df[trace + "_sedw"],
                color=colors[i],
                alpha=alphafill,
            )
        else:
            if se:
                ax.plot(
                    df[trace],
                    color=colors[i],
                    alpha=alphas[i],
                    label=trace,
                    linewidth=2,
                )

    ax.set_ylabel("Normalized Vm")
    # ax.text(0.8, 0.9, 'CP-ISO', ha='left', va='center',
    #         transform=ax.transAxes, size='large')

    # fontname = 'Arial', fontsize = 14)
    ax.set_xlabel("Relative Time (ms)")
    for ax in fig.get_axes():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()
    ax.set_xlim(-90, 65)

    ax.set_ylim(-0.15, 1.4)
    ax.axhline(0, alpha=0.2, color="k")

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
    y = df.loc[0, [traces[0]]]
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


plt.close("all")

if "popspeed_df" not in dir():
    popspeed_df = load_pop_datafile(key="speed", display=True)
anot = True
fig = plot_speed(popspeed_df, substract=False, spread=[0, 1], anot=True)

# save = False
# if save:
#     file = "f10_speed"
#     paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
#     for ext in [".png", ".pdf", ".svg"]:
#         file_name = os.path.join(paths["save"], (file + ext))
#         fig.savefig(file_name)
save = False
if save:
    folder = paths["figSup"]
    file = "f11_speed"
    for ext in [".pdf"]:
        filename = os.path.join(folder, file + ext)
        fig.savefig(filename)
    folder = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        filename = os.path.join(folder, (file + ext))
        fig.savefig(filename)
