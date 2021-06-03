#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:11:03 2021

@author: cdesbois
"""

from datetime import datetime
import os
import xml.etree.ElementTree as ET

# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd

import config

paths = config.build_paths()


dirname = os.path.join(paths["owncFig"], "data", "gabyToCg", "sources")

file_name = os.path.join(dirname, "4100gg3_vm_cp_cx_max.svg")
# filename = os.path.expanduser('~/4100gg3_vm_cp_cx_max.svg')

file_name = os.path.join(dirname, "test.svg")

#%%
# with open(file_name, 'r') as f:
#     data = f.read()

# Bs_data = BeautifulSoup(data, 'xml')

# # b_unique = Bs_data.find_all()
# # print(b_unique)


# -> a metadata

# nb properties in [tag, attributes, text string, tail string, child element]


def extract_data(filename):

    tree = ET.parse(filename)
    roots = tree.getroot()
    print(roots)

    for root in roots:
        print(list(root))

    labels = set()

    gs = []
    lines = []
    polylines = []

    root = roots[1]
    for item in list(root):
        # check if no child
        if not list(item):
            print("no elements for {}".format(item.tag))
            continue
        # list child
        for ob in list(item):
            # print('='*20)
            # print(ob)
            label = ob.tag.split("}")[-1]
            labels.add(label)
            if label == "g":
                gs.append(ob.attrib)
            elif label == "line":
                lines.append(ob.attrib)
            elif label == "polyline":
                polylines.append(ob.attrib)
            else:
                print("{} is not a referenced label".format(label))
            # print('label = {}'.format(label))
            # print('contains for {}'.format(ob.tag))
            # #  print('attrib= {}'.format(ob.attrib))
            # print('txt= {}'.format(ob.text))
            # print('list= {}'.format(list(ob)))
            # print('-'*20)
    print("=" * 20)
    print("founded {} labels".format(labels))
    for label, val in zip(sorted(list(labels)), [gs, lines, polylines]):
        print("{} {}".format(len(val), label))
    return gs, lines, polylines


gs, lines, polylines = extract_data(file_name)


#%%
item = polylines[0]
pt = item["points"]
temp = [_ for _ in pt.strip().split(" ")]
temp = [_.split(",") for _ in temp]
temp = [(float(a), float(b)) for a, b in temp]
temp = list(set(temp))

#%%

plt.close("all")

fig = plt.figure()
ax = fig.add_subplot(111)

for item in polylines:
    ax.plot(item["points"], label=item["class"])

#%%


def plot_cgGabyVersion(datadf):
    
    ncols = 2
    nrows = 1

    inset_hfrac = .3
    inset_vfrac = .3

    inset_hfrac_offset = .6
    inset_vfrac_offset = .6

    top_pad = .1
    bottom_pad = .1
    left_pad = .1
    right_pad = .1
    
    hspace = .1
    vspace = .1

    ax_width = (1 - left_pad - right_pad - (ncols - 1) * hspace) / ncols
    ax_height = (1 - top_pad - bottom_pad - (nrows - 1) * vspace) / nrows

    fig = plt.figure(figsize=(14, 8))

    ax_lst = []
    for j in range(ncols):
        for k in range(nrows):
            a_bottom = bottom_pad + k * ( ax_height + vspace)
            a_left = left_pad + j * (ax_width + hspace)
        
            inset_bottom = a_bottom + inset_vfrac_offset * ax_height
            inset_left = a_left + inset_hfrac_offset * ax_width

            ax = fig.add_axes([a_left, a_bottom, ax_width, ax_height])
            ax_in = fig.add_axes([inset_left, inset_bottom, ax_width * inset_hfrac, ax_height *  inset_vfrac])
            ax_lst.append((ax,ax_in))


    
    # fig, axes = plt.subplots(figsize=(14, 8), nrows=1, ncols=2, 
    #                          sharex=True, sharey=True)
    # axes = axes.flatten()

    # # rect = l, b, w, h
    # anchor = [0.3, 0.6] 
    # size = [0.4, 0.2]
    # fig.add_axes(anchor + size, facecolor='y', alpha=0.3)
    # anchor[0] += 0.5
    # fig.add_axes(anchor + size, facecolor='y', alpha=0.3)
    
    # axes = ax_lst

    df = datadf.copy()
    middle = (df.index.max() - df.index.min()) / 2
    df.index = (df.index - middle) / 10
    # limit the date time range
    df = df.loc[-200:200]
    std_colors = config.std_colors()
    colors = [
        std_colors["red"],
        std_colors["red"],
        std_colors["yellow"],
        std_colors["yellow"],
        "k",
    ]
    style = ["-", ":", "-", ":", "-"]
    linewidth = [2, 3, 2, 3, 2]
    alpha = [0.8, 0.8, 1, 1, 0.7]
    # left right
    for i, cell in enumerate(cells[::-1]):
        # if i == 0:
        #     continue
        # main plot
        ax = ax_lst[i][0]
        # cell = cells[0]
        cols = [_ for _ in df.columns if cell in _]
        labels = [_.split("_")[2:] for _ in cols]
        labels = [[st.title() for st in _] for _ in labels]
        labels = ["".join(st) for st in labels]
        labels = [_.replace("CtrCtr", "") for _ in labels]

        for j, col in enumerate(cols):
            ax.plot(
                df[col],
                linestyle=style[j],
                linewidth=linewidth[j],
                color=colors[j],
                alpha=alpha[j],
                label=labels[j],
            )
            # ax.plot(df[cols], label=labels)
            ax.set_title(cell)
        #insert 
        ax = ax_lst[i][1]
        for j, col in enumerate(cols):
            ax.plot(
                df[col],
                linestyle=style[j],
                linewidth=linewidth[j],
                color=colors[j],
                alpha=alpha[j],
                label=labels[j],
            )
    # to adapt to gaby
    for axs in ax_lst:
        ax = axs[0]
        ax.set_xlim(-42.5, 206)
        ax.set_ylim(-5, 15)
        ax.set_yticks(range(0, 14, 2))
        ax = axs[1]
        ax.set_xlim(-42.5, 206)
        ax.set_ylim(-5, 15)
    
    for i, ax in enumerate(fig.get_axes()):
        ax.axhline(y=0, alpha=0.5, color="k")
        ax.axvline(x=0, alpha=0.5, color="k")
        ax.set_xlabel("Time (ms)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if not i%2: # only main
            ax.legend()
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

    return fig


plt.close("all")
anot = True

file = "cg_specificity.xlsx"
file_name = os.path.join(dirname, file)
data_df = pd.read_excel(file_name)
cells = list(set([_.split("_")[0] for _ in data_df.columns]))

fig = plot_cgGabyVersion(data_df)


save = False
if save:
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "gabyToCg")
    file_name = os.path.join(dirname, "cgGabyVersion.png")
    fig.savefig(file_name)


#%%
plt.close('all')

fig, axes = plt.subplots(figsize=(14, 8), nrows=1, ncols=2, 
                             sharex=True, sharey=True)
axes = axes.flatten()
yx = lambda x: 20/(206 + 42.5)
# rect = l, b, w, h
anchor = [0.25, 0.5] 
size = [0.2, 0.3]
fig.add_axes(anchor + size, facecolor='y', alpha=0.3)
anchor[0] += 0.5
fig.add_axes(anchor + size, facecolor='y', alpha=0.3)
#ax.set_aspect(0.8)

#%%
plt.close('all')

