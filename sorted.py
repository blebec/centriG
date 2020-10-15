#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot centrigabor sorted responses
"""

import os
from importlib import reload
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import config
import general_functions as gfunc
import load.load_data as ldat


#===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])

#%% plot latency (left) and gain (right)

plt.close('all')

def plot_all_sorted_responses(overlap=True, sort_all=True, key=0,
                               spread='sect',
                               kind='vm', age='old', amp='engy'):
    """
    plot the sorted cell responses
    input = conditions parameters
    overlap : boolean, overlap the different rows to superpose the plots
    sort_all : if false, only the 'key' trace is sorted
    key : number to choose the trace to be sorted
    output : matplotlib plot
    """
    def set_ticks_both(axis):
        """ set ticks and ticks labels on both sides """
        ticks = list(axis.majorTicks) # a copy
        ticks.extend(axis.minorTicks)
        for t in ticks:
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
            t.label1.set_visible(True)
            t.label2.set_visible(True)
    titles = {'engy' : r'$\Delta$ energy',
              'time50' : 'latency advance',
              'gain50' : 'amplitude gain',
              'sect' : 'sector',
              'spk' : 'spikes',
              'vm' : 'vm', 
              'full': 'full'}

    # parameter
    cols = [std_colors[item] \
              for item in ['red', 'green', 'yellow', 'blue', 'blue']]
    colors = []
    for item in zip(cols, cols):
        colors.extend(item)
    # data (call)
    df = ldat.load_cell_contributions(rec=kind, amp=amp, age=age)
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if spread in item]
    #remove the 'rdsect'
    traces = [item for item in traces if 'rdisosect' not in item]
    # append full random
    if not 'rdisofull' in [item.split('_')[0] for item in traces]:
        rdfull = [item for item in df.columns if 'rdisofull' in item]
        traces.extend(rdfull)
    # filter -> only significative cells
    traces = [item for item in traces if not item.endswith('sig')]
    # text labels
    title = '{} {}'.format(titles.get(kind, ''), titles.get(spread, ''))
    anotx = 'Cell rank'
    if age == 'old':
        anoty = [r'$\Delta$ Phase (ms)', r'$\Delta$ Amplitude']
             #(fraction of Center-only response)']
    else:
        anoty = [titles['time50'], titles.get(amp, '')]
    # plot
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True,
                             sharey='col', squeeze=False)#•sharey=True,
    if anot:
        fig.suptitle(title, alpha=0.4)
    axes = axes.flatten()
    x = range(1, len(df)+1)
    # use cpisotime for ref
    name = traces[0]
    name = traces[key]
    sig_name = name + '_sig'
    df = df.sort_values(by=[name, sig_name], ascending=False)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + '_sig'
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0 : 'w', 1 : edge_color}
        if sort_all:
            select = df[[name, sig_name]].sort_values(by=[name, sig_name],
                                                      ascending=False)
        else:
            select = df[[name, sig_name]]
        bar_colors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        # ax.set_title(str(i))
        ax.bar(x, select[name], color=bar_colors, edgecolor=edge_color,
               alpha=0.8, width=0.8)
        if i in [0, 1]:
            ax.set_title(anoty[i])
    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[::2]
    right_axes = axes[1::2]
    for axe in [left_axes, right_axes]:
        for i, ax in enumerate(axe):
            ax.set_facecolor('None')
            # ax.set_title(i)
            ax.spines['top'].set_visible(False)
            ax.ticklabel_format(useOffset=True)
            ax.spines['bottom'].set_visible(False)
            # zero line
            ax.axhline(0, alpha=0.3, color='k')
            if i != 4:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel(anotx)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xticks([1, len(df)])
                ax.set_xlim(0, len(df)+1)
    for ax in left_axes:
        custom_ticks = np.linspace(0, 10, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    for ax in right_axes:
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
    no_spines = True
    if no_spines == True:
        for ax in left_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 10, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
        for ax in right_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.axvline(limx[1], 0, -0.5, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)

    # align each row yaxis on zero between subplots
    gfunc.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gfunc.change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots
    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.5, wspace=0.2)
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'sorted.py:plot_all_sorted_responses',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

#%% old
# fig = plot_sorted_responses_sup1(overlap=True)
# fig = plot_sorted_responses_sup1(overlap=True, sort_all=False)
#%%
kind = ['vm', 'spk'][0]

fig = plot_all_sorted_responses(overlap=True, sort_all=False,
                                 kind=kind, amp='engy', age='new')

fig = plot_all_sorted_responses(overlap=True, sort_all=True,
                                 kind=kind, amp='engy', age='new')


fig = plot_all_sorted_responses(overlap=True, sort_all=False, key=1,
                                 kind=kind, amp='engy', age='new')
#%%
plt.close('all')
save = True
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'sorted', 'sorted&contrib')
amp = 'engy'
for kind in ['vm', 'spk']:
    for spread in ['sect', 'full']:
    # for amp in ['gain', 'engy']:
        figs = []
        figs.append(plot_all_sorted_responses(overlap=True, sort_all=True,
                                               kind=kind, amp=amp, age='new',
                                               spread=spread))
        # figs.append(plot_sorted_responses_sup1(overlap=True, sort_all=False,
        #                                         kind=kind, amp=amp, age='new'))
        # figs.append(plot_sorted_responses_sup1(overlap=True, sort_all=False, key=1,
        #                                         kind=kind, amp=amp, age='new'))
        if save:
            for i, fig in enumerate(figs):
                filename = os.path.join(paths['save'], kind + spread.title() + '_' + amp + str(i) + '.png')
                fig.savefig(filename, format='png')


# =============================================================================
# savePath = os.path.join(paths['cgFig'], 'pythonPreview', 'sorted', 'testAllSortingKeys')
# for key in range(10):
#     fig = plot_sorted_responses_sup1(overlap=True, sort_all=False, key=key)
#     filename = os.path.join(savePath, str(key) + '.png')
#     fig.savefig(filename, format='png')

# =============================================================================