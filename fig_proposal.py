#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:19:26 2020

@author: cdesbois
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

import plot_general_functions as gfuc
import load_data as ldat
    
anot=True


def plot_2B_bis(stdColors, anot=False):
    """
    plot_figure2B alternative : sorted phase advance and delta response
    response are sorted only by phase
    """
    df = ldat.load_cell_contributions('vm')
    alist = [item for item in df.columns if 'vm_s_cp_iso_' in item]
    df = df[alist].sort_values(by=alist[0], ascending=False)
    cols = df.columns[::2]
    sigs = df.columns[1::2]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    color_dic = {0 :'w', 1 : stdColors['rouge']}
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[sigs[i]]]
        axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                    color=colors, label=cols[i], alpha=0.8, width=0.8)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        ax.set_xticks([0, len(df)-1])
        ax.set_xticklabels([1, len(df)])
        ax.set_xlim(-1, len(df)+0.5)
        if i == 0:
            ax.vlines(-1, 0, 20, linewidth=2)
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
            ylabel = r'$\Delta$ Phase (ms)'
            ax.set_ylim(-6, 29)
            x_label = 'Cell rank'
        else:
            ax.vlines(-1, 0, 0.6, linewidth=2)
            custom_yticks = np.linspace(0, 0.6, 4)
            x_label = 'Ranked cells'
            ylabel = r'$\Delta$ Amplitude'
        ax.set_xlabel(x_label)
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(ylabel)
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_visible(False)

    gfuc.align_yaxis(axes[0], 0, axes[1], 0)
    gfuc.change_plot_trace_amplitude(axes[1], 0.8)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B_bis',
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
        fig =  plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ref, '-k', alpha = 0.5, label='ref')
        ax.plot(cp, '-r', alpha = 0.6, label='cp')
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


def plot_figure6_bis(stdColors, linear=True, substract=False):
    """
    plot_figure6 minus center
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = ['center_only', 'surround_then_center', 'surround_only',
            'static_linear_prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'], stdColors['vertSombre']]
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
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    # vlocs = np.linspace(-0.7, -1.7, 4)
    # vlocs = np.linspace(-1.4, -2.4, 4)
    vlocs = np.linspace(-0.8, -1.5, 4)
    dico = dict(zip(names, hlocs))

    # ax
    for key in dico.keys():
        # names
        ax.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                    annotation_clip=False, fontsize='small')
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
        ax.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                    annotation_clip=False, fontsize='small')

    ax.set_ylabel('Membrane potential (mV)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # response start
    x = 41
    y = df['center_only'].loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    ax.vlines(x, -0.5, lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    for dloc in hlocs:
        ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    # end
    x = 150.1
    ax.vlines(x, -0.5, lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    # peak
    x = 63.9
    ax.vlines(x, -0.5, lims[1], 'k', alpha=0.5)
    ax.axvspan(41, 150.1, ymin=0.28, color='k', alpha=0.1)
    # ticks
    custom_ticks = np.linspace(0, 1, 2, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.55, 0.17, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure6_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

#%%
def plot_figure7_bis(stdColors):
    """
    plot_figure7 minus center
    """
    filename = 'data/fig6.xlsx'
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
    colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'],
              stdColors['bleuViolet'], stdColors['bleuViolet'],
              stdColors['bleuViolet']]
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'],
              stdColors['bleuViolet'], stdColors['rouge'],
              stdColors['bleuViolet']]
    alphas = [0.5, 0.7, 0.7, 0.6, 0.6, 0.6]

    # plotting
    fig = plt.figure(figsize=(8.5, 4))
    ax = fig.add_subplot(111)
    ax.plot(df.popfillVmscpIsoDlp, ':r', alpha=1, linewidth=2,
            label='sThenCent - cent')
    # surroundOnly
    ax.plot(df.surroundOnlysdUp, color=stdColors['vertSombre'], alpha=0.7,
            label='surroundOnly')
    ax.fill_between(df.index, df[df.columns[3]], df[df.columns[4]],
                    color=stdColors['vertSombre'], alpha=0.2)

    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=12", xy=(0.1, 0.8),
                xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.3)
    # response start
    x0 = 0
    y = df['centerOnly'].loc[x0]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.vlines(x0, lims[0], lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    # end
    x2 = 124.6
    y = df['centerOnly'].loc[x2]
    ax.vlines(x2, lims[0], lims[1], color='k',
              linestyle='-', alpha=0.2)
    # ax.plot(x2, y, 'o', color=stdColors['bleu'])
    # peak
    # df.centerOnly.idxmax()
    x1 = 26.1
    ax.vlines(x1, 0, lims[1], 'k', alpha=0.3)
    ax.axvspan(x0, x2, color='k', alpha=0.1)
    #ticks
    custom_ticks = np.linspace(0, 0.4, 3)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.5, 0.10, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure7_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig