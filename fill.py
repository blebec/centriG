#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12 nov 2020 15:07:38 CET
@author: cdesbois
"""


import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra
import fig_proposal as figp


anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])

paths = config.build_paths()
paths['save'] = os.path.join(paths['owncFig'],
                             'pythonPreview', 'fillinIn', 'indFill_popFill')

# load data
indi_df = ldat.load_filldata('indi')
pop_df = ldat.load_filldata('pop')

#%%
def plot_indFill(data, stdcolors=std_colors, anot=True):
    """
    plot_figure6
    """
    df = data.copy()
    cols = ['Center-Only', 'Surround-then-Center',
            'Surround-Only', 'Static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = [stdcolors[st] for st in
              ['k', 'red', 'dark_green', 'dark_green']]
    alphas = [0.6, 0.8, 0.8, 0.8]

    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True,
                             figsize=(8.5, 8))
    axes = axes.flatten()
    # fig.suptitle(os.path.basename(filename))
    # traces
    ax = axes[0]
    for i, col in enumerate(cols[:2]):
        ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                 label=col)
    ax.spines['bottom'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    ax = axes[1]
    for i, col in enumerate(cols):
        if i == 3:
            ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                     label=col, linestyle='--', linewidth=1.5)
        else:
            ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                     label=col)
    ax.set_xlabel('Time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    # vlocs = np.linspace(-0.7, -1.7, 4)
    vlocs = np.linspace(-1.4, -2.4, 4)
    dico = dict(zip(names, hlocs))

    ax = axes[0]
    for key in dico.keys():
        # name
        ax.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax.add_patch(rect)
    # center
    rect = Rectangle(xy=(0, vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])#'k')
    ax.add_patch(rect)

    st = 'Surround-then-Center'
    ax.annotate(st, xy=(30, vlocs[1]), color=colors[1], alpha=1,
                 annotation_clip=False, fontsize='small')
    st = 'Center-Only'
    ax.annotate(st, xy=(30, vlocs[2]), color=colors[0], alpha=1,
                 annotation_clip=False, fontsize='small')
        # see annotation_clip=False
    ax.set_ylim(-2.5, 4.5)

    # ax2
    ax = axes[1]
    for key in dico.keys():
        # names
        ax.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=1, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                             fill=True, alpha=1, edgecolor=colors[2],
                             facecolor='w')
        ax.add_patch(rect)
        # stim2
        rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax.add_patch(rect)
    #center
    rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax.add_patch(rect)
    for i, st in enumerate(['Surround-Only', 'Surround-then-Center', 'Center-Only']):
        ax.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                     annotation_clip=False, fontsize='small')
    for ax in axes:
        # leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
        #                handlelength=0)
        # colored text
        # for line, text in zip(leg.get_lines(), leg.get_texts()):
            # text.set_color(line.get_color())
        ax.set_ylabel('Membrane potential (mV)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        ax.axhline(0, alpha=0.2, color='k')
        ax.axvline(0, alpha=0.2, color='k')
        # response start
        x = 41
        y = df['Center-Only'].loc[x]
        ax.plot(x, y, 'o', color='tab:blue', ms=10, alpha=0.8)
        ax.vlines(x, -1, 2, color='tab:blue', linestyle=':', alpha=0.8)
        for dloc in hlocs:
            ax.axvline(dloc, linestyle=':', alpha=0.2, color='k')
        #ticks
        custom_ticks = np.linspace(0, 4, 5, dtype=int)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fill.py:plot_indFill',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plt.close('all')
# indi_df = load_filldata('indi')
fig = plot_indFill(indi_df, std_colors, anot=anot)
save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                           'pythonPreview', 'fillingIn', 'indFill_popFill')
    file_name = os.path.join(dirname, 'indFill.png')
    fig.savefig(file_name)

#%%
plt.close('all')

fig = figp.plot_indFill_bis(indi_df, std_colors)
# fig = plot_figure6_bis(substract=True)
fig = figp.plot_indFill_bis(indi_df, std_colors, linear=False, substract=True)

#%%

def plot_pop_predict(data, lp='minus', stdcolors=std_colors):
    """
    plot_figure7
    lP <-> linear predictor
    """
    df = data.copy()
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
            'sosdUp', 'sosdDown', 'solinearPrediction', 'stcsdUp',
            'stcsdDown', 'stcLinearPreediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = [stdcolors[st] for st in
              ['k', 'red', 'dark_green', 'blue_violet', 'blue_violet',
               'blue_violet', 'red', 'red', 'blue_violet']]
    alphas = [0.5, 0.7, 0.7, 0.6, 0.6, 0.5, 0.2, 0.2, 0.7]

    sharesY = dict(minus = False, plus = True)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                             sharey=sharesY[lp], figsize=(8.5, 8))
    axes = axes.flatten()

    ax = axes[0]
    cols = df.columns[:3]
    linewidths = (1, 1, 2)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
        linewidth=linewidths[i], label=col)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    vspread = .06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')

    lims = dict(minus = (-0.2, 1.1), plus=(-0.05, 1.2))
    ax.set_ylim(lims.get(lp))

    # predictive magnification
    ax = axes[1]
    colors = [stdcolors[st]
              for st in ['k', 'red', 'dark_green', 'blue_violet']]
    linewidths=(1,1)
    # (first, second, stdup, stddown)
    lp_cols = dict(minus=[2, 5, 3, 4], plus=[1, 6, 7, 8])
    cols = [df.columns[i] for i in lp_cols[lp]]
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i+2], alpha=alphas[i+2],
                label=col, linewidth=linewidths[i])
    ax.fill_between(df.index, df[cols[2]], df[cols[3]],
                    color=colors[2], alpha=0.1)

    for i, ax in enumerate(axes):
        ax.axhline(0, alpha=0.3, color='k')
        ax.axvline(0, linewidth=2, color='tab:blue', linestyle=':')
        ax.set_ylabel('Normalized membrane potential')
        ax.annotate("n=12", xy=(0.1, 0.8),
                    xycoords="axes fraction", ha='center')
        if i > 0:
            ax.set_xlabel('Relative time (ms)')
            # if lp == 'minus':
            #     gfunc.change_plot_trace_amplitude(ax, 0.9)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()

    if lp == 'minus':
        custom_ticks = np.arange(0, 1.1, 0.2)
        axes[0].set_yticks(custom_ticks)
    elif lp =='plus':
        custom_ticks = np.arange(0, 1.2, 0.2)
        axes[0].set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:pop_predict',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plt.close('all')

fig = plot_pop_predict(pop_df, 'minus')
#fig = plot_pop_predict('plus')
fig2 = figp.plot_pop_fill_bis(pop_df, std_colors)
save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                           'pythonPreview', 'fillingIn', 'indFill_popFill')
    file_name = os.path.join(dirname, 'predict_Fill.png')
    fig.savefig(file_name)

#%%

def plot_pop_fill(data, stdcolors=std_colors, anot=anot):
    """
    plot_figure7
    lP <-> linear predictor
    """
    df = data.copy()
    cols = gfunc.new_columns_names(df.columns)
    cols = ['_'.join(item.split('_')[1:]) for item in cols]
    df.columns = cols

    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly',
            'sosdUp', 'sosdDown', 'solinearPrediction', 'stcsdUp',
            'stcsdDown', 'stcLinearPrediction',
            'stcvmcfIso', 'stcvmcpCross', 'stcvmfRnd', 'stcvmsRnd',
            'stcspkcpCtr, stcspkcpIso',
            'stcspkcfIso', 'stcspkcpCross','stcspkfRnd', 'stcspksRnd']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = [stdcolors[st] for st in
              ['k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]

    spks = cols[13:17]
    vms = [df.columns[i] for i in [0, 1, 9, 10, 11]]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 13),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # vm pop
    ax = axes[0]
    for i, col in enumerate(vms):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    ax.set_xlim(-20,50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=???", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    # spk pop
    ax = axes[1]
    for i, col in enumerate(spks):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    x = 0
    y = df[spks[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-20,60)
    ax.set_ylabel('Normalized firing rate')
    ax.annotate("n=7", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.axhline(0, alpha=0.3, color='k')
        ax.axvline(0, linewidth=2, color='tab:blue', linestyle=':')
        custom_ticks = np.arange(0, 1.1, 0.2)
        ax.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fill.py:plot_pop_fill',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')

fig = plot_pop_fill(pop_df, std_colors, anot)
save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                           'pythonPreview', 'fillingIn', 'indFill_popFill')
    file = os.path.join(dirname, 'pop_fill.png')
    fig.savefig(os.path.join(dirname, file))

#%%

def plot_pop_fill_2X2(df, lp='minus', stdcolors=std_colors):
    """
    plot_figure7
    lP <-> linear predictor
    """

    cols = gfunc.new_columns_names(df.columns)
    cols = ['_'.join(item.split('_')[1:]) for item in cols]
    # df.columns = cols

    # cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
    #         'sosdUp', 'sosdDown', 'solinearPrediction', 'stcsdUp',
    #         'stcsdDown', 'stcLinearPrediction',
    #         'stcvmcfIso', 'stcvmcpCross', 'stcvmfRnd', 'stcvmsRnd',
    #         'stcspkcpCtr, stcspkcpIso',
    #         'stcspkcfIso', 'stcspkcpCross','stcspkfRnd', 'stcspksRnd']
    # dico = dict(zip(df.columns, cols))
    # df.rename(columns=dico, inplace=True)
    # colors = ['k', std_colors['red'], std_colors['dark_green'],
    #           std_colors['blue_violet'], std_colors['blue_violet'],
    #           std_colors['blue_violet'], std_colors['red'],
    #           std_colors['red'], std_colors['blue_violet'],
    #           std_colors['green'], std_colors['yellow'],
    #           std_colors['blue'], std_colors['blue'],
    #           'k', std_colors['red'],
    #           std_colors['green'], std_colors['yellow'],
    #           std_colors['blue'], std_colors['blue']]
    colors = [stdcolors[st] for st in
              ['k', 'red', 'dark_green',
              'blue_violet', 'blue_violet', 'blue_violet',
              'red', 'red', 'blue_violet', 'green', 'yellow', 'blue',
              'blue',
              'k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.5, 0.7, 0.7,
              0.5, 0.5,
              0.6, 0.2,
              0.2, 0.7,
              0.7, 0.7,
              0.7, 0.7,
              0.5, 0.7,
              0.7, 0.7,
              0.7, 0.7]

    fig = plt.figure(figsize=(11.6, 10))
    # fig.suptitle(os.path.basename(filename))

    # traces & surround only
    ax0 = fig.add_subplot(221)
    cols = df.columns[:3]
    linewidths = (1, 1, 2)
    for i, col in enumerate(cols):
        ax0.plot(df[col], color=colors[i], alpha=alphas[i],
                 linewidth=linewidths[i], label=col)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax0.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    lims = dict(minus = (-0.2, 1.1), plus=(-0.05, 1.2))
    ax0.set_ylim(lims.get(lp))

    # surroundOnly & predictor
    if lp == 'minus':
        ax1 = fig.add_subplot(223, sharex=ax0)
    elif lp =='plus':
        ax1 = fig.add_subplot(223, sharex=ax0, sharey=ax0)
    colors = [stdcolors[st]
              for st in ['k', 'red', 'dark_green', 'blue_violet']]
    linewidths=(2,1)
    # (first, second, stdup, stddown)
    lp_cols = dict(minus=[2, 5, 3, 4], plus=[1, 6, 7, 8])
    cols = [df.columns[i] for i in lp_cols[lp]]
    for i, col in enumerate(cols[:2]):
        ax1.plot(df[col], color=colors[i+2], alpha=alphas[i+2],
                label=col, linewidth=linewidths[i])
    ax1.fill_between(df.index, df[cols[2]], df[cols[3]],
                    color=colors[2], alpha=0.1)

    # populations
    colors = [stdcolors[st] for st in
              ['k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]
    spk_cols = df.columns[13:18]
    vm_cols = [df.columns[i] for i in [0, 1, 9, 10, 11]]
    # vm
    ax2 = fig.add_subplot(222, sharey= ax0)
    for i, col in enumerate(vm_cols):
        ax2.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    ax2.set_xlim(-20,50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax2.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax2.set_ylabel('Normalized membrane potential')
    ax2.annotate("n=???", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')

    # spk
    ax3 = fig.add_subplot(224, sharex= ax2)
    for i, col in enumerate(spk_cols):
        ax3.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    x = 0
    y = df[spk_cols[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax3.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax3.set_ylabel('Normalized firing rate')
    ax3.annotate("n=7", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax3.set_xlabel('Relative time (ms)')


    ax0.annotate("n=12", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax0.set_ylabel('Normalized membrane potential')
    ax2.set_ylabel('Normalized membrane potential')
    ax1.set_xlabel('Relative time (ms)')
    ax3.set_xlabel('Relative time (ms)')
    ax3.set_ylabel('Normalized firing rate')
    ax3.annotate("n=7", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')

    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        ax.axhline(0, alpha=0.3, color='k')
        ax.axvline(0, linewidth=2, color='tab:blue', linestyle=':')
        # ax.axvline(0, alpha=0.3, color='k')
    # align zero between subplots
    #gfunc.align_yaxis(ax1, 0, ax2, 0)
    if lp == 'minus':
        gfunc.change_plot_trace_amplitude(ax2, 0.9)
    fig.tight_layout()
    # add ref
    ref = (0, df.loc[0, [df.columns[0]]])

    if lp == 'minus':
        custom_ticks = np.arange(0, 1.1, 0.2)
        ax0.set_yticks(custom_ticks)
    elif lp =='plus':
        custom_ticks = np.arange(0, 1.2, 0.2)
        ax1.set_yticks(custom_ticks)
        ax3.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fill.py:plot_pop_fill_2X2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plt.close('all')
#fig1 = plot_pop_fill_2X2(pop_df, 'plus', std_colors)
fig2 = plot_pop_fill_2X2(pop_df, 'minus', std_colors)
save = False
if save:
    # file = 'pop_fill_2X2_plus.png'
    # fig1.savefig(os.path.join(paths['save'], file))
    file = 'pop_fill_2X2_minus.png'
    fig2.savefig(os.path.join(paths['save'], file))

#%%

def plot_pop_fill_surround(data, stdcolors=std_colors):
    """
    plot_figure7 surround only vm responses

    """
    df = data.copy()
    cols = gfunc.new_columns_names(df.columns)
    cols = ['_'.join(item.split('_')[1:]) for item in cols]
    df.columns = cols

    # fig = plt.figure(figsize=(11.6, 10))
    fig = plt.figure(figsize=(6.5, 5.5))
   # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    surround_cols = [df.columns[st] for st in (2,19,20,21)]
    colors = [stdcolors[st] for st in
              ['k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.5, 0.7, 0.7, 0.5, 0.5, 0.6]
    # +1 because no black curve
    for i, col in enumerate(surround_cols):
    # for i in (2,19,20,21):
        ax.plot(df[col], color=colors[i+1], alpha=alphas[i+1],
                 linewidth=2, label=col)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    #vspread = .06  # vertical spread for realign location
    # ax1.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-150,150)

    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=12", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')

    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.axhline(0, alpha=0.3, color='k')
        ax.axvline(0, linewidth=2, color='tab:blue', linestyle=':')
    # align zero between subplots
    # gfunc.align_yaxis(ax1, 0, ax2, 0)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fill.py:plot_pop_fill_surround',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plt.close('all')

fig = plot_pop_fill_surround(pop_df, std_colors)

# # to use with pop_subtraction
# ax = fig.get_axes()[0]
# ax.set_xlim(-150, 150)
# ax.set_ylim(-0.15, 0.35)
# ax.set_xticks(np.linspace(-150, 150, 7))
# ax.set_yticks(np.linspace(-0.1, 0.3, 5))

save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                           'pythonPreview', 'fillingIn', 'indFill_popFill')
    file = 'pop_fill_surround.png'
    fig.savefig(os.path.join(dirname, file))

#%% plot combi
plt.close('all')
def plot_fill_combi(data_fill, data_pop, stdcolors=std_colors, anot=anot):

    colors = [stdcolors[st] for st in 
              ['k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    # fill pop
    df = data_fill.copy()
    
    # general  pop
    gen_df = data_pop.copy()
    #defined in dataframe columns (first column = ctr))
    kind, rec, spread,  *_ = gen_df.columns.to_list()[1].split('_')
    # centering
    middle = (gen_df.index.max() - gen_df.index.min())/2
    gen_df.index = (gen_df.index - middle)/10
    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    # subtract the centerOnly response (ref = df['CENTER-ONLY'])
    ref = gen_df[gen_df.columns[0]]
    gen_df = gen_df.subtract(ref, axis=0)
    # remove rdsect
    cols = gen_df.columns.to_list()
    while any(st for st in cols if 'sect_rd' in st):
        cols.remove(next(st for st in cols if 'sect_rd' in st))
    #buils labels
    labels = cols[:]
    labels = [n.replace('full_rd_', 'full_rdf_') for n in labels]
    for i in range(3):
        for item in labels:
            if len(item.split('_')) < 6:
                j = labels.index(item)
                labels[j] = item + '_ctr'
    labels = [st.split('_')[-3] for st in labels]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17, 17))
    axes = axes.flatten('F')
    for i, ax in enumerate(axes):
        ax.set_title(i)

    # fill pop
    spks = df.columns[13:17]
    vms = [df.columns[i] for i in [0, 1, 9, 10, 11]]

    # vm pop
    ax = axes[0]
    for i, col in enumerate(vms):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    ax.set_xlim(-20,50)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=???", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')

    # spk pop
    ax = axes[1]
    for i, col in enumerate(spks):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label= df.columns[i])
    x = 0
    y = df[spks[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    vspread = .06  # vertical spread for realign location
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-20,50)
    ax.set_ylabel('Normalized firing rate')
    ax.annotate("n=7", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')

    # gen population
    ax = axes[2]
    cols = gen_df.columns
    for i, col in enumerate(cols[:-1]):
        ax.plot(gen_df[col], color=colors[i], alpha=alphas[i], label=labels[i],
                linewidth=2)
    # bluePoint
    # x = 0
    # y = df.loc[0][df.columns[0]]
    # vspread = .02  # vertical spread for realign location
    # # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
    # ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    # ax.axvline(x, linewidth=2, color='tab:blue', linestyle=':')
    # ax.plot(0, df.loc[0][df.columns[0]], 'o', color=colors[0],
    #         ms=10, alpha=0.5)
    #labels
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
  
    # max_x center only
    ax.axvline(21.4, alpha=0.4, color='k')
    # end_x of center only
    #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
    ax.axvline(88, alpha=0.3)
    ax.axvspan(0, 88, facecolor='k', alpha=0.2)

    ax.text(0.45, 0.9, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    ax.set_ylabel('Norm Vm - Norm centerOnly')

    # surround only
    ax = axes[3]
    surround_cols = [df.columns[st] for st in (2,19,20,21)]
    # +1 because no black curve
    for i, col in enumerate(surround_cols):
    # for i in (2,19,20,21):
        ax.plot(df[col], color=colors[i+1], alpha=alphas[i+1],
                 linewidth=2, label=col)
    # response point
    x = 0
    y = df[df.columns[0]].loc[0]
    # ax1.plot(x, y, 'o', color=std_colors['blue'])
    #vspread = .06  # vertical spread for realign location
    # ax1.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.set_xlim(-150,150)
    
    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=12", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.axhline(0, alpha=0.3, color='k')
        ax.axvline(0, linewidth=2, color='tab:blue', linestyle=':')
    # align zero between subplots
    # gfunc.align_yaxis(ax1, 0, ax2, 0)
    for ax in axes[:2]:
        ax.set_xlim(-20, 60)
        custom_ticks = np.arange(-20, 60, 10)[1:]
        ax.set_xticks(custom_ticks)
        ax.set_ylim(-.05, 1.1)
        custom_ticks = np.arange(0, 1.1, 0.2)
        ax.set_yticks(custom_ticks)
                  
    for ax in axes[2:]:
        ax.set_xlim(-150, 150)
        ax.set_ylim(-0.15, 0.35)
        ax.set_xticks(np.linspace(-150, 150, 7)[1:-1])
        ax.set_yticks(np.linspace(-0.1, 0.3, 5))
        
        
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fill.py:plot_fill_combi',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


plt.close('all')
# general pop
select = dict(age='new', rec='vm', kind='sig')
# select['align'] = 'p2p'

data_df, file = ltra.load_intra_mean_traces(paths, **select)

plot_fill_combi(pop_df, data_df)