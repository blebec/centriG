#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:27:33 2020

@author: cdesbois
"""
import os

import pandas as pd
import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import config

def plot_cellDepth():
    """
    relation profondeur / latence
    """
    # cells
    data50 = ld.load_50vals('vm')
    # retain only the neuron names
    data50.reset_index(inplace=True)
    data50.Neuron = data50.Neuron.apply(lambda x: x.split('_')[0])
    data50.set_index('Neuron', inplace=True)
    # layers (.csv file include decimals in dephts -> bugs)
    filename = os.path.join(paths['owncFig'], 'cells/centri_neurons_histolog_170105.xlsx')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    # lay_df = pd.concat([data50, df])
    lay_df = pd.concat([data50, df], axis=1, join='inner')
    labelled = lay_df.dropna(subset=['CDLayer']).copy()
    labelled.CDLayer = labelled.CDLayer.astype('int')

    colors = {6:'k', 5:'b', 4:'g', 3:'r'}
    fig = plt.figure()
    fig.suptitle('cp iso  : layers effect')
    ax = fig.add_subplot(211)
    kind = 'cpisosect_time50'
    labelled = labelled.sort_values(by=kind, ascending=False)
    y = labelled.cpisosect_time50.to_list()
    x = labelled.index.to_list()
    z = [colors[item] for item in labelled.CDLayer.to_list()]
    # stat
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    ax.bar(x, y, color=z2, edgecolor=z1, linewidth=3,
           alpha=0.6, width=0.8)
    # ax.bar(x, y, color = z, alpha=0.6)
    ax.set_ylabel('advance (ms)')
    ax.set_xlabel('cell')
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)

    ax = fig.add_subplot(212)
    kind = 'cpisosect_gain50'
    y = labelled[kind].to_list()
    # stat
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    ax.bar(x, y, color=z2, edgecolor=z1, linewidth=3,
           alpha=0.6, width=0.8)

    # ax.bar(x, y, color = z, alpha=0.6)
    ax.set_ylabel('gain')
    ax.set_xlabel('cell')
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)

    # leg = 'black= layer 6, \n blue= layer 5, \n green= layer4, \n red=layer 3'
    axes = fig.get_axes()
    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, len(x))
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i == 0:
            # ax.text(0.8, 0.7, leg, transform=ax.transAxes)
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.text(0.80, 0.95, 'layer 3', color='r', transform=ax.transAxes)
            ax.text(0.80, 0.85, 'layer 4', color='g', transform=ax.transAxes)
            ax.text(0.80, 0.75, 'layer 5', color='b', transform=ax.transAxes)
            ax.text(0.80, 0.65, 'layer 6', color='k', alpha=0.6, transform=ax.transAxes)
            custom_ticks = np.linspace(0, 10, 2, dtype=int)
            ax.set_yticks(custom_ticks)
            ax.set_yticklabels(custom_ticks)
            ax.vlines(ax.get_xlim()[0], 0, 10, linewidth=2)
            ax.spines['left'].set_visible(False)
        else:
            ax.tick_params(axis='x', labelrotation=45)
            custom_ticks = np.linspace(0, 0.4, 2)
            ax.set_yticks(custom_ticks)
            ax.set_yticklabels(custom_ticks)
            ax.vlines(ax.get_xlim()[0], 0, 0.4, linewidth=2)
            ax.spines['left'].set_visible(False)

        fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_cellDepth',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


# TODO add the significativity for each condition
# ? grou by depth

def plot_cellDepth_all(spread='sect'):
    """
    relation profondeur / latence
    """
    # cells
    data50 = ld.load_50vals('vm')
    # retain only the neuron names
    data50.reset_index(inplace=True)
    data50.Neuron = data50.Neuron.apply(lambda x: x.split('_')[0])
    data50.set_index('Neuron', inplace=True)
    # layers (.csv file include decimals in dephts -> bugs)
    filename = os.path.join(paths['owncFig'], 'cells/centri_neurons_histolog_170105.xlsx')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    # lay_df = pd.concat([data50, df])
    lay_df = pd.concat([data50, df], axis=1, join='inner')
    labelled = lay_df.dropna(subset=['CDLayer']).copy()
    kind = 'cpisosect_time50'
    labelled.CDLayer = labelled.CDLayer.astype('int')
    # sort by response
    labelled = labelled.sort_values(by=kind, ascending=False)
    # an the sort by depth
    labelled = labelled.sort_values(by='CDLayer', ascending=True)
    colors = {6:'k', 5:'b', 4:'g', 3:'r'}
    x = labelled.index.to_list()
    y = labelled[kind].to_list()
    z = [colors[item] for item in labelled.CDLayer.to_list()]
    # stat z1 = list(depht_colors), z2 = list(white if non significant)
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    # figure
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 8), sharex=True)
    axes = axes.T.flatten().tolist()
    # shareY
    axr = axes[0]
    for ax in axes[0::2]:
        ax.get_shared_y_axes().join(ax, axr)
    axr = axes[1]
    for ax in axes[1::2]:
        ax.get_shared_y_axes().join(ax, axr)
    # fill axes
    # 'sect' or 'full'
    alist = [item for item in labelled.columns if spread in item]
    # remove separate the sig column
    val_list = [item for item in alist if '_sig' not in item]
    sig_list = [item for item in alist if '_sig' in item]
    # choose advance values
    cols = [item for item in val_list if 'time50' in item]
    sig_cols = [item for item in sig_list if 'time50' in item]
    for ax, col, sig_col in zip(axes[0::2], cols, sig_cols):
        ax.set_title(col.split('_')[0])
        ax.set_ylabel('advance')
        y = labelled[col].to_list()
        #stat z1 = list(depht_colors), z2 = list(white if non significant)
        z1 = z.copy()
        z2 = []
        for a, b in zip(z1, labelled[sig_col].to_list()):
            if b == 1:
                z2.append(a)
            else:
                z2.append('w')
        # ax.bar(x, y, color=z, alpha=0.6)
        ax.bar(x, y, color=z2, edgecolor=z1, linewidth=2,
               alpha=0.6, width=0.8)
        ax.set_xlabel('cell')
        ax.set_xlim(-0.5, len(x))
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        custom_ticks = np.linspace(0, 10, 2)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
        ax.vlines(ax.get_xlim()[0], 0, 10, linewidth=2)
        ax.spines['left'].set_visible(False)
    # choose gain values
    cols = [item for item in val_list if 'gain50' in item]
    sig_cols = [item for item in sig_list if 'gain50' in item]
    for ax, col, sig_col in zip(axes[1::2], cols, sig_cols):
        ax.set_title(col.split('_')[0])
        ax.set_ylabel('gain')
        y = labelled[col].to_list()
        # stat z1 = list(depht_colors), z2 = list(white if non significant)
        z1 = z.copy()
        z2 = []
        for a, b in zip(z1, labelled[sig_col].to_list()):
            if b == 1:
                z2.append(a)
            else:
                z2.append('w')
        # ax.bar(x, y, color = z, alpha=0.6)
        ax.bar(x, y, color=z2, edgecolor=z1, linewidth=2,
               alpha=0.6, width=0.8)
        ax.set_xlabel('cell')
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
        ax.vlines(ax.get_xlim()[0], 0, 0.5, linewidth=2)
        ax.spines['left'].set_visible(False)
    for i, ax in enumerate(axes):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i in [0, 4]:
            pass
            # ax.text(0.7, 0.7, leg, transform=ax.transAxes)
        if i not in [3, 7]:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.tick_params(axis='x', labelrotation=45)
    for ax in [axes[0], axes[4]]:
        ax.text(0.08, 0.85, 'layer 3', color='r', transform=ax.transAxes)
        ax.text(0.22, 0.85, 'layer 4', color='g', transform=ax.transAxes)
        ax.text(0.45, 0.85, 'layer 5', color='b', transform=ax.transAxes)
        ax.text(0.85, 0.85, 'layer 6', color='k', alpha=0.6, transform=ax.transAxes)
        ax.margins(0.01)
    fig.subplots_adjust(hspace=0.02)
    fig.subplots_adjust(wspace=0.2)
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_cellDepth_all',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig



if __name__ == '__main__':
    paths = config.build_paths()
    anot = True
    fig = plot_cellDepth()
    fig1 = plot_cellDepth_all(spread='sect')
    fig2 = plot_cellDepth_all(spread='full')
