#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:19:26 2020

@author: cdesbois
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import plot_general_functions as gf
import load_data as ld


def plot_2B_bis(stdColors, anot=False):
    """
    plot_figure2B alternative : sorted phase advance and delta response
    response are sorted only by phase
    """
    df = ld.load_cell_contributions('vm')
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

    gf.align_yaxis(axes[0], 0, axes[1], 0)
    gf.change_plot_trace_amplitude(axes[1], 0.8)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

