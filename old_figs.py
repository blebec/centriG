#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:36:09 2020

@author: cdesbois
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plot_general_functions as gfuc
from datetime import datetime
from matplotlib.ticker import StrMethodFormatter


def plot_2_indMoySigNsig(data, colsdict, stdColors, fill=True, anot=False):
    """
    plot_figure2 (individual + moy + sig + nonsig)
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]

    #fig = plt.figure(figsize=(inchtocm(17.6),inchtocm(12)))
    fig = plt.figure(figsize=(16.6, 12))
    #build axes with sharex and sharey
    axes = []
    axL = fig.add_subplot(2, 4, 1)
    axes.append(axL)
    ax = fig.add_subplot(2, 4, 2)
    axes.append(ax)
    axes.append(fig.add_subplot(2, 4, 3, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 4, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 5, sharex=axL))
    ax = fig.add_subplot(2, 4, 6)
    axes.append(ax)
    axes.append(fig.add_subplot(2, 4, 7, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 8, sharex=ax, sharey=ax))
    # axes list
    vmaxes = axes[:4]      # vm axes = top row
    spkaxes = axes[4:]     # spikes axes = bottom row
    #____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alphas[i],
                label=col)
    # start point
    x = 41.5
    y = data.indiVmctr.loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'])
    # invert the plot order for spikes
    inv_colors = colors[::-1]
    inv_alphas = alphas[::-1]
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
        ax.plot(data[col], color=inv_colors[i],
                alpha=1, label=col)  #, linewidth=1)
#    ax.plot(39.8, 0.1523, 'o', color= stdColors['bleu'])
    x = 39.8
    y = data.indiSpkCtr.loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'])
    # plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[2]
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        #errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alphas[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                                label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[3]
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        #errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alphas[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                                label=col, linewidth=0.5)
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=inv_colors[i],
                alpha=1, label=col)#, linewidth=1)
        # ax.fill_between(df.index, df[col],
        #                 color=inv_colors[i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

# TODO define a Spk plotmode[lines, allhist, sdFill] for popSpkSig and popSpkNsig
    #popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=inv_alphas[i]/2)# label=col, linewidth=0.5)
        # for j in [0, 1]:
        #     ax.plot(df[col[j]], color=colors[i],
        #             alpha=1, label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[3]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col[0]], df[col[1]], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2)#, label=col, linewidth=0.5)
        # for j in [0, 1]:
        #     ax.plot(df[col[j]], color=colors[i],
        #             alpha=1, label=col, linewidth=0.5)
    ax.annotate("n=15", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['Membrane potential (mV)',
               'Normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['Firing rate (spikes/s)',
               'Normalized firing rate',
               '', '']
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('time (ms)')

    for ax in vmaxes[1:]:
        ax.set_ylim(-0.10, 1.2)
    for ax in spkaxes[1:]:
        ax.set_ylim(-0.10, 1.3)
        ax.set_xlabel('relative time (ms)')

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    dico = dict(zip(names, xlocs))
    # lines
    for ax in [vmaxes[0], spkaxes[0]]:
        lims = ax.get_ylim()
        for dloc in xlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    # stim location
    ax = spkaxes[0]
    for key in dico.keys():
        ax.annotate(key, xy=(dico[key]+3, -3), alpha=0.6, fontsize='x-small')
        # stim
        rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
                         alpha=0.6, edgecolor='w', facecolor=stdColors['rouge'])
        ax.add_patch(rect)
        # center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    # fit individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    gfuc.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    gfuc.align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    gfuc.change_plot_trace_amplitude(vmaxes[1], 0.85)
    gfuc.change_plot_trace_amplitude(spkaxes[1], 0.8)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    # rebuild tick to one decimal
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 12, 8, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    custom_ticks = np.linspace(0, 15, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    ax = vmaxes[1]
    custom_ticks = np.linspace(-0.2, 1, 7)
    ax.set_yticks(custom_ticks)
    ax = spkaxes[1]
    custom_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(custom_ticks)

    fig.tight_layout()

    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:old_figs.plot_2_indMoySigNsig',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def plot_2_indMoy(data, colsdict, stdColors, anot=False):
    """
    plot_2 indiv + pop
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]

    fig = plt.figure(figsize=(8.5, 8))
    # build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    # plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alphas[i],
                label=col)
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.plot(data[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    # plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
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
    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['Membrane potential (mV)',
               'Normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['Firing rate (spikes/s)',
               'Normalized firing rate',
               '', '']
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('Time (ms)')

    for ax in vmaxes[1:]:
        ax.set_ylim(-0.10, 1.1)
    for ax in spkaxes[1:]:
        ax.set_ylim(-0.10, 1.1)
        ax.set_xlabel('Relative time (ms)')

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    dico = dict(zip(names, xlocs))
    # lines
    for ax in [vmaxes[0], spkaxes[0]]:
        lims = ax.get_ylim()
        for dloc in xlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    # stim location
    ax = spkaxes[0]
    for key in dico.keys():
        ax.annotate(key, xy=(dico[key]+3, -3), alpha=0.6, fontsize='x-small')
        # stim
        rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
                         alpha=0.6, edgecolor='w', facecolor=stdColors['rouge'])
        ax.add_patch(rect)
        # center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    # fit individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    gfuc.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    gfuc.align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    gfuc.change_plot_trace_amplitude(vmaxes[1], 0.85)
    gfuc.change_plot_trace_amplitude(spkaxes[1], 0.8)
    # adjust ticks
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 12, 8, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    custom_ticks = np.linspace(0, 15, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    ax = vmaxes[1]
    custom_ticks = np.arange(-0.2, 1.2, 0.2)
    ax.set_yticks(custom_ticks)
    ax = spkaxes[1]
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_2_indMoy',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def plot_2_sigNsig(data, colsdict, stdColors, fill=True, 
                           fillground=True, anot=False):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(8.5, 8)) #fig = plt.figure(figsize=(8, 8))
    # build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    # popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[-2]
    ax.set_title('significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alphas[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                            label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[-1]
    ax.set_title('non significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alphas[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                            label=col, linewidth=0.5)
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # NB for spiking response, the control should be plotted last,
    # ie revert the order
    inv_colors = colors[::-1]
    inv_alphas = alphas[::-1]
    # popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[-2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2, label=col)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[-1]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2)
    ax.annotate("n=15", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['membrane potential (mV)',
               'normalized membrane potential',
               '', '']
    ylabels = ylabels[1:]   # no individual exemple
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['firing rate (spikes/s)',
               'normalized firing rate',
               '', '']
    ylabels = ylabels[1:]   # no individual exemple
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('time (ms)')

    for ax in vmaxes:
        ax.set_ylim(-0.10, 1.2)
    for ax in spkaxes:
        ax.set_ylim(-0.10, 1.3)
        ax.set_xlabel('relative time (ms)')
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_2_sigNsig',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def plot_3_signonsig(stdColors, anot=False, substract=False):
    """
    plot_figure3
    with individually significants and non significant cells
    """
    filenames = {'pop' : 'data/fig3.xlsx',
                 'sig': 'data/fig3bis1.xlsx',
                 'nonsig': 'data/fig3bis2.xlsx'}
    titles = {'pop' : 'recorded cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    alphas = [0.8, 1, 0.8, 0.8, 0.8]

    fig = plt.figure(figsize=(11.6, 6))
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(1, 2, i+1))
    for i, kind in enumerate(['sig', 'nonsig']):
        ncells = cellnumbers[kind]
        df = pd.read_excel(filenames[kind])
        # centering
        middle = (df.index.max() - df.index.min())/2
        df.index = (df.index - middle)/10
        df = df.loc[-45:120]
        cols = ['CNT-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
        df.columns = cols
        if substract:
            ref = df['CNT-ONLY']
            df = df.subtract(ref, axis=0)
        ax = axes[i]
        ax.set_title(titles[kind])
        ncells = cellnumbers[kind]
        for j, col in enumerate(cols):
            ax.plot(df[col], color=colors[j], alpha=alphas[j],
                    label=col, linewidth=2)
            ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
                        xycoords="axes fraction", ha='center')
        if substract:
            leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
                            handlelength=0)
        else:
            leg = ax.legend(loc='lower right', markerscale=None, frameon=False,
                            handlelength=0)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        # bleu point
        ax.plot(0, df.loc[0]['CNT-ONLY'], 'o', color=stdColors['bleu'])

    axes[0].set_ylabel('normalized membrane potential')
    for ax in axes:
        ax.set_xlim(-15, 30)
        ax.set_ylim(-0.1, 1.1)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ax.set_xlabel('relative time (ms)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    if substract:
        for ax in axes:
            ax.set_xlim(-45, 120)
            ax.set_ylim(-0.15, 0.4)
            custom_ticks = np.arange(-40, 110, 20)
            ax.set_xticks(custom_ticks)
            # max center only
            lims = ax.get_ylim()
            ax.vlines(21.4, lims[0], lims[1], alpha=0.5)
            # end of center only
            #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
            ax.vlines(88, lims[0], lims[1], alpha=0.3)
            ax.axvspan(0, 88, facecolor='k', alpha=0.1)
            ax.text(0.41, 0.6, 'center only response \n start | peak | end',
                    transform=ax.transAxes, alpha=0.5)
        axes[0].set_ylabel('Norm vm - Norm centerOnly')
        # axes[1].yaxis.set_visible(False)
        # axes[1].spines['left'].set_visible(False)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_3_signonsig',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig
