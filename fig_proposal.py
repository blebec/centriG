#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:19:26 2020

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

anot = True
std_colors = config.std_colors()

paths = config.build_paths()
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'proposal')

#%%
plt.close('all')

def plot_figure2(datadf, colsdict, anot=False, age='new'):
    """
    figure2 (individual + pop + sig)
    """
    colors = ['k', std_colors['red']]
    alphas = [0.8, 0.8]

    fig = plt.figure(figsize=(17.6, 12))
    axes = []
    ax= fig.add_subplot(221)
    axes.append(ax)
    axes.append(fig.add_subplot(223, sharex=ax))
    ax= fig.add_subplot(222)
    axes.append(ax)
    axes.append(fig.add_subplot(224, sharex=ax, sharey = ax))

    # axes list
    vmaxes = (axes[0], axes[2])      # vm axes = top row
    spkaxes = (axes[1], axes[3])     # spikes axes = bottom row
    # ____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(datadf[col], color=colors[i], alpha=alphas[i],
                label=col)
    # response point
    if age == 'old':
        x = 41.5
    else:
        x = 43.5
    y = datadf.indiVmctr.loc[x]
    #blue point and vline
    ax.plot(x, y, 'o', color=std_colors['k'], ms=10, alpha=0.5)
    # lims = ax.get_ylim()´
    ax.axvline(x, linewidth=2, color=std_colors['k'],
              linestyle=':', alpha = 0.5)
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.plot(datadf[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(datadf.index, datadf[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    # response point
    if age == 'old':
        x = 39.8
    else:
        x = 55.5
    y = datadf.indiSpkCtr.loc[x]
    ax.plot(x, y, 'o', color=std_colors['k'], ms=10, alpha=0.5)
    ax.axvline(x, linewidth=2, color=std_colors['k'],
              linestyle=':')
    ax.axvline(x, linewidth=2, color=std_colors['k'],
              linestyle=':')
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    # ____ plots pop (column 1-3)
    df = datadf.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # # popVmSig
    # cols = colsdict['popVmSig']
    # ax = vmaxes[2]
    # # traces
    # for i, col in enumerate(cols[:2]):
    #     ax.plot(df[col], color=colors[i], alpha=alphas[i],
    #             label=col)
    #     # errors : iterate on tuples
    #     for i, col in enumerate(cols[2:]):
    #         ax.fill_between(df.index, df[col[0]], df[col[1]],
    #                         color=colors[i], alpha=0.2)#alphas[i]/2)
    # # advance
    # x0 = 0
    # y = df.loc[x0][cols[0]]
    # adf = df.loc[-20:0, [cols[1]]]
    # i1 = (adf - y).abs().values.flatten().argsort()[0]
    # x1 = adf.index[i1]
    # ax.plot(x0, y, 'o', color=std_colors['blue'], ms=10, alpha=0.8)
    # ax.plot(x1, y, marker=markers.CARETLEFT, color=std_colors['blue'],
    #         ms=10, alpha=0.8)
    # lims = ax.get_ylim()
    # # ax.vlines(y, *lims, color=std_colors['blue'], linestyle=':', linewidth=2)
    # ax.hlines(y, x0, x1, color=std_colors['blue'], linestyle=':', linewidth=2)
    # ax.annotate("n=10", xy=(0.2, 0.8),
    #             xycoords="axes fraction", ha='center')
    # # adv = str(x0 - x1)
    # # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
    #             #xycoords="axes fraction", ha='center')
    # pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=colors[::-1][i],
                alpha=1, label=col)#, linewidth=1)
        # ax.fill_between(df.index, df[col],
        #                 color=colors[::-1][i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # # popSpkSig
    # cols = colsdict['popSpkSig']
    # ax = spkaxes[2]
    # # traces
    # for i, col in enumerate(cols[:2][::-1]):
    #     ax.plot(df[col], color=colors[::-1][i], alpha=1, label=col)
    # # errors : iterate on tuples
    # for i, col in enumerate(cols[2:]):
    #     ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
    #                     alpha=alphas[::-1][i]/2)# label=col, linewidth=0.5)
    # # advance
    # x0 = 0
    # y = df.loc[x0][cols[0]]
    # adf = df.loc[-20:0, [cols[1]]]
    # i1 = (adf - y).abs().values.flatten().argsort()[0]
    # x1 = adf.index[i1]
    # ax.plot(x0, y, 'o', color=std_colors['blue'], ms=10, alpha=0.8)
    # # ax.plot(x0, y, 'o', color=std_colors['blue'])
    # ax.plot(x1, y, marker=markers.CARETLEFT, color=std_colors['blue'],
    #         ms=10, alpha=0.8)
    # # ax.axvline(y, color=std_colors['blue'], linestyle=':', linewidth=2)
    # ax.hlines(y, x0, x1, color=std_colors['blue'], linestyle=':', linewidth=2)

    # ax.annotate("n=5", xy=(0.2, 0.8),
    #             xycoords="axes fraction", ha='center')
    # # #advance
    # # adv = str(x0 - x1)
    # # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
    #             # xycoords="axes fraction", ha='center')
    # labels
    ylabels = ['Membrane potential (mV)',
               'Firing rate (spk/sec)',
               'Normalized membrane potential',
               'Normalized firing rate']
    for i, ax in enumerate(axes):
        ax.axvline(0, alpha=0.4)
        ax.axhline(0, alpha=0.4)
        ax.set_ylabel(ylabels[i])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i % 2 == 0:
            ax.spines['bottom'].set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
        axes[1].set_xlabel('Time (ms)')
        axes[3].set_xlabel('Relative time (ms)')
        # normalized y
        axes[1].set_ylim(-0.10, 1.1)

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ['D' + str(i) for i in range(6)]
    dico = dict(zip(names, xlocs))
    # lines
    for ax in axes[:2]:
        lims = ax.get_ylim()
        for dloc in xlocs:
            ax.axvline(dloc, linestyle=':', alpha=0.4, color='k')
    # fit individual example
    if age == 'old':
        vmaxes[0].set_ylim(-3.5, 12)
        spkaxes[0].set_ylim(-5.5, 18)
        # stim location
        ax = spkaxes[0]
        for key in dico.keys():
            ax.annotate(key, xy=(dico[key]+step/2, -3), alpha=0.6,
                        ha='center', fontsize='x-small')
            # stim
            rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
                             alpha=0.6, edgecolor='w', facecolor=std_colors['red'])
            ax.add_patch(rect)
        # center
        rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                         alpha=0.6, edgecolor='w', facecolor='k')
        ax.add_patch(rect)
    else:
        axes[0].set_ylim(-3.5, 12)
        axes[1].set_ylim(-10, 35)
        # stim location
        ax = axes[1]
        for key in dico.keys():
            ax.annotate(key, xy=(dico[key]+step/2, -5), alpha=0.6,
                        ha='center', fontsize='x-small')
            # stim
            rect = Rectangle(xy=(dico[key], -9), width=step, height=2, fill=True,
                             alpha=0.6, edgecolor='w', facecolor=std_colors['red'])
            ax.add_patch(rect)
        # center
        rect = Rectangle(xy=(0, -7), width=step, height=2, fill=True,
                         alpha=0.6, edgecolor='w', facecolor='k')
        ax.add_patch(rect)

    # align zero between plots  NB ref = first plot
    gfunc.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    # adjust amplitude (without moving the zero)
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 10, 7, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    if  age == 'old':
        custom_ticks = np.linspace(0, 15, 4, dtype=int)
    else:
        custom_ticks = np.linspace(0, 30, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    custom_ticks = np.linspace(0, 1, 6)
    vmaxes[1].set_yticks(custom_ticks)
    custom_ticks = np.linspace(0, 1, 6)
    spkaxes[1].set_yticks(custom_ticks)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06, wspace=0.4)
    # align ylabels
    fig.align_ylabels()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

#data
age = ['old', 'new'][1]
fig2_df, fig2_dict = ldat.load2(age)
fig = plot_figure2(datadf=fig2_df, colsdict=fig2_dict, anot=anot, age=age)
save = False
if save:
    filename = os.path.join(paths['save'], 'prop_fig2.png')
    fig.savefig(filename)
#%%
# 
def plot_2B_bis(stdcolors, anot=False, age='new'):
    """
    plot_figure2B alternative : sorted phase advance and delta response
    response are sorted only by phase
    """
    amp = 'engy' # 'gain'
    df = ldat.load_cell_contributions(rec='vm', amp=amp, age=age)
    traces = [item for item in df.columns if item.startswith('cpisosect')]
    df = df[traces].sort_values(by=traces[0], ascending=False)
    vals = [item for item in df.columns if not item.endswith('_sig')]
    sigs = [item for item in df.columns if item.endswith('_sig')]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    color_dic = {0 :'w', 1 : stdcolors['red']}
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[sigs[i]]]
        label = vals[i].split('_')[1]
        axes[i].bar(df.index, df[vals[i]], edgecolor=stdcolors['red'],
                    color=colors, label=label, alpha=0.8, width=0.8)
        lims = ax.get_xlim()
        ax.axhline(0, *lims, alpha=0.4)
        ax.set_xticks([0, len(df)-1])
        ax.set_xticklabels([1, len(df)])
        ax.set_xlim(-1, len(df)+0.5)
        if i == 0:
            ax.vlines(-1, 0, 20, linewidth=2)
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
            ylabel = r'$\Delta$ Phase (ms)'
            ax.set_ylim(-6, 29)
            if amp == 'engy':
                ax.set_ylim(-10, 29)    
            x_label = 'Cell rank'
        else:
            ax.vlines(-1, 0, 0.6, linewidth=2)
            custom_yticks = np.linspace(0, 0.6, 4)
            x_label = 'Ranked cells'
            if amp == 'gain':
                ylabel = r'$\Delta$ Amplitude'
            else:
                ylabel = r'$\Delta$ Energy'                
        ax.set_xlabel(x_label)
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(ylabel)
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_visible(False)

    gfunc.change_plot_trace_amplitude(axes[1], 0.8)
    gfunc.align_yaxis(axes[0], 0, axes[1], 0)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fig_proposal.py:plot_figure2B_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig



#%%

def align_center(adf, showPlot=False):
    """
    align the traces on the center response to build figure 6
    """
    df = adf.copy()
    ref = df['center_only'].copy()
    cp = df.surround_then_center.copy()
    ref50_y = (ref.loc[30:80].max() - ref.loc[30:80].min()) / 2
    ref50_x = (ref.loc[30:80] - ref50_y).abs().sort_values().index[0]
    cp50_y = ref50_y
    cp50_x = ((cp.loc[30:70] - cp50_y)).abs().sort_values().index[0]
    adv = cp50_x - ref50_x
    print('adv=', adv)
    ref_corr = ref.shift(int(10*adv))
    if showPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ref, '-k', alpha=0.5, label='ref')
        ax.plot(cp, '-r', alpha=0.6, label='cp')
        ax.set_xlim(0, 100)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        limx = ax.get_xlim()
        limy = ax.get_ylim()
        ax.hlines(ref50_y, limx[0], limx[1], alpha=0.3)
        ax.vlines(ref50_x, limy[0], limy[1], alpha=0.3)
        ax.vlines(cp50_x, limy[0], limy[1], 'r', alpha=0.4)
        ax.plot(ref.shift(int(10*adv)), ':k', alpha=0.5, label='ref_corr',
                linewidth=2)
        ax.plot(cp - ref, '-b', alpha=0.5, label='cp - ref')
        ax.plot(cp - ref_corr, ':b', alpha=0.5, label='cp - ref_corr',
                linewidth=2)
        ax.legend()
        fig.tight_layout()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    return ref_corr


def plot_figure6_bis(stdcolors, linear=True, substract=False):
    """
    plot_figure6 minus center
    """
    filename = 'data/data_to_use/indifill.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = ['center_only', 'surround_then_center', 
            'surround_only', 'static_linear_prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdcolors['red'], 
              stdcolors['dark_green'], stdcolors['dark_green']]
    alphas = [0.6, 0.8, 0.8, 0.8]
    # substract
    # build a time shifted reference (centerOnly) to perfome the substraction
    ref_shift = align_center(df, showPlot=False)         ### CALL
    df['surround_then_center'] = df['surround_then_center'] - ref_shift
   # df['Center_Only'] -= ref
    # plotting
    fig = plt.figure(figsize=(8.5, 4))
#    fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols): # omit centrerOnly
        if i == 0:
            pass
        # suround the center minus center
        elif i == 1:
            if substract:
                ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                        label=col, linestyle='--', linewidth=1.5)
        # linear predictor
        elif i == 3:
            if linear:
                ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                        label=col, linestyle='--', linewidth=1.5)
        else:
            ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=1,
                    label=col)
    ax.set_xlabel('Time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    
    names = ['D' + str(i) for i in range(6)]
    # vlocs = np.linspace(-0.7, -1.7, 4)
    # vlocs = np.linspace(-1.4, -2.4, 4)
    vlocs = np.linspace(-0.8, -1.5, 4)
    dico = dict(zip(names, hlocs))

    # ax
    for key in dico.keys():
        # names
        ax.annotate(key, xy=(dico[key]+step/2, vlocs[0]), alpha=0.6,
                    annotation_clip=False, fontsize='small', ha='center')
        #stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.2,
                         fill=True, alpha=1, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.2,
                             fill=True, alpha=1, edgecolor=colors[2],
                             facecolor='w')
        ax.add_patch(rect)
        # #stim2
        # rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
        #                  fill=True, alpha=0.6, edgecolor='w',
        #                  facecolor=colors[1])
        # if key == 'D0':
        #     rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
        #                  fill=True, alpha=0.6, edgecolor='k',
        #                  facecolor='y')
        # ax.add_patch(rect)
    # #center
    # rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
    #                  alpha=0.6, edgecolor='w', facecolor=colors[0])
    # ax.add_patch(rect)
#    for i, st in enumerate(['Surround-Only', 'Surround-then-Center minus Center']):
    for i, st in enumerate(['surround-only']):
        ax.annotate(st, xy=(30, vlocs[i+1]+0.05), color=colors[2-i],
                    annotation_clip=False, fontsize='small', va='bottom')
    ax.set_ylabel('Membrane potential (mV)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    #zero lines and stim limist
    # ax.axhline(0, *lims, alpha=0.3)
    # lims = ax.get_ylim()
    # ax.axvline(0, *lims, alpha=0.3)
    ax.axhline(0, alpha=0.4)
    ax.axvline(0, alpha=0.4)
    for dloc in hlocs:
        ax.axvline(dloc, linestyle=':', alpha=0.3)
    # response
    lims = ax.get_ylim()
    lims = (-0.5, lims[1])  # to avoid overlap with the legend
    #start
    x = 41
    y = df['center_only'].loc[x]
    ax.plot(x, y, 'o', color=stdcolors['blue'], ms=10, alpha=0.8)
    ax.vlines(x, *lims, color=stdcolors['blue'],
              linestyle=':', alpha=0.8)
    # end
    x = 150.1
    ax.vlines(x, *lims, color=stdcolors['blue'],
              linestyle=':', alpha=0.8)
    # peak
    x = 63.9
    ax.vlines(x, *lims, 'k', alpha=0.5)
    # y in ax coordinates (ie in [0,1])
    _, y0 = gfunc.axis_data_coords_sys_transform(ax, 150.1, 0, True)
    ax.axvspan(41, 150.1, y0, 1 , color='k', alpha=0.1)
    # ticks
    custom_ticks = np.linspace(0, 1, 2, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.55, 0.17, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fig_proposal.py:plot_figure6_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

#%%
def plot_figure7_bis(stdcolors):
    """
    plot_figure7 minus center
    """
    filename = 'data/data_to_use/popfill.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # limit the date time range
    df = df.loc[-150:160]
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
            'sdUp', 'sdDown', 'linearPrediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    # if substract:
    #     ref = df.centerOnly
    #     df.surroundThenCenter = df.surroundThenCenter - ref
    #     df.centerOnly = df.centerOnly - ref
    #colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = [std_colors[item] for item in \
             ['k', 'red', 'dark_green', 'blue_violet', 'red', 'blue_violet']]    
    alphas = [0.5, 0.7, 0.7, 0.6, 0.6, 0.6]

    # plotting
    fig = plt.figure(figsize=(8.5, 4))
    ax = fig.add_subplot(111)
    ax.plot(df.popfillVmscpIsoMDlp, ':r', alpha=1, linewidth=2,
            label='sThenCent - cent')
    # surroundOnly
    ax.plot(df.surroundOnlysdUp, color=stdcolors['dark_green'], alpha=0.7,
            label='surroundOnly')
    ax.fill_between(df.index, df[df.columns[3]], df[df.columns[4]],
                    color=stdcolors['dark_green'], alpha=0.2)

    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=12", xy=(0.1, 0.8),
                xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    ax.axhline(0, alpha=0.3)
    ax.axvline(0, alpha=0.3)
    # response start
    lims = (0, ax.get_ylim()[1])
    x0 = 0
    y = df['centerOnly'].loc[x0]
    ax.plot(x0, y, 'o', color=stdcolors['blue'])
    ax.vlines(x0, *lims, color=stdcolors['blue'],
              linestyle=':', alpha=0.8)
    # end
    x2 = 124.6
    y = df['centerOnly'].loc[x2]
    ax.vlines(x2, *lims, color='k',
              linestyle='-', alpha=0.2)
    # ax.plot(x2, y, 'o', color=stdcolors['blue'])
    # peak
    # df.centerOnly.idxmax()
    x1 = 26.1
    ax.vlines(x1, *lims, 'k', alpha=0.3)
    # build shaded area : x in data coordinates, y in figure coordinates
    _, y2 = gfunc.axis_data_coords_sys_transform(ax, 0, 0, True)
    ax.axvspan(x0, x2, y2, 1, color='k', alpha=0.1)
    #ticks
    custom_ticks = np.linspace(0, 0.4, 3)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.5, 0.10, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'fig_proposal.py:plot_figure7_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


if __name__ == "__main__":
    anot = True
    std_colors = config.std_colors()
    plot_2B_bis(std_colors, anot=anot)
    plot_figure6_bis(std_colors, linear=True, substract=False)
    plot_figure6_bis(std_colors, linear=True, substract=True)
    plot_figure7_bis(std_colors)
#TODO append the save process