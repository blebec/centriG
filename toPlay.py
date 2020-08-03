#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:24:28 2020

@author: cdesbois
"""

from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import config
import load_data as ldat
std_colors = config.std_colors()
anot = True

#%% a test fig and no more
plt.close('all')

fig = plt.figure()
axes = []
ax = fig.add_subplot(421)
axes.append(ax)
for i in range(3, 8, 2):
    print(i)
    axes.append(fig.add_subplot(4, 2, i))
for i in range(2, 9, 2):
    axes.append(fig.add_subplot(4, 2, i))

fig, axes = plt.subplots(nrows=4, ncols=2)
axes = axes.T.flatten().tolist()
axr = axes[0]
for ax in axes[0::2]:
    ax.get_shared_y_axes().join(ax, axr)
axr = axes[1]
for ax in axes[1::2]:
    ax.get_shared_y_axes().join(ax, axr)

#%% explore peaks and gain

def load_peakdata(name):
    'load the excel file'
    # name = 'data/cg_peakValueTime_spk.xlsx'
    df = pd.read_excel(name)
    # replace 'sec' by 'sect' for homogeneity
    new_list = []
    for item in df.columns:
        new_list.append(item.replace('sec', 'sect'))
    df.columns = new_list
    # adapt column names
    new_list = []
    for item in df.iloc[0].tolist():
        if 'value' in str(item):
            new_list.append('_gainP')
        elif 'time' in str(item):
            new_list.append('_timeP')
        else:
            new_list.append('')
    cols = [item.split('.')[0] for item in df.columns]
    cols = [a + b for a, b in zip(cols, new_list)]
    df.columns = cols
    # remove first line
    df = df.drop(df.index[0])
    # remove empty column
    df = df.drop('Unnamed: 10', axis=1)
    df = df.set_index('Neuron')
    df = df.astype('float')
    #rename
    cols = df.columns
    cols = [item.replace('ctr', 'centeronly') for item in cols]
    cols = [item.replace('rnd', 'rd') for item in cols]
    cols = [item.replace('rdfull', 'rdisofull') for item in cols]
    cols = [item.replace('rdsec', 'rdisosec') for item in cols]
    cols = [item.replace('cross', 'crx') for item in cols]
    cols = [item.replace('isosfull', 'isofull') for item in cols]
    df.columns = cols
    return df


def normalize_peakdata_and_select(df, spread='sect', param='gain'):
    """
    return the normalized and selected df parts for plotting
    spread in ['sect', 'full'],
    param in ['time', 'gain']
    """
    if spread not in ['sect', 'full']:
        print("'spread' should be in ['sect', 'full']")
        return
    if param not in ['time', 'gain']:
        print("'param' should be in ['time', 'gain']")
        return
    # select by param (first value = control)
    col_list = [item for item in df.columns if param in item]
    # normalization with center only (Y - Yref)/Yref
    ctrl = df[col_list[0]]
    for item in col_list[1:]:
        # print(df[item].mean())
        df[item] = (df[item] - ctrl) / ctrl
        # print(df[item].mean())
    # select by spread
    col_list = [item for item in col_list if spread in item]
    return df[col_list]


def plot_sorted_responses(df_left, df_right, mes='', overlap=True,
                          left_sig=True, right_sig=True):
    """
    plot the sorted cell responses
    input = dataframes, overlap=boolean

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

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anot_left = df_left.columns[0].split('_')[1]
    anot_right = df_right.columns[0].split('_')[1]
    # anoty = ['Relative peak advance(ms)', 'Relative peak amplitude']
    #           #(fraction of Center-only response)']
    # plot
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2)
    # left
    left_axes = []
    ax = fig.add_subplot(gs[0, 0])
    left_axes.append(ax)
    for i in range(1, 4):
        left_axes.append(fig.add_subplot(gs[i, 0],
                                         sharey=ax, sharex=ax))
    # right
    right_axes = []
    ax = fig.add_subplot(gs[0, 1])
    right_axes.append(ax)
    for i in range(1, 4):
        right_axes.append(fig.add_subplot(gs[i, 1],
                                          sharey=ax, sharex=ax))
    # to identify the plots (uncomment to use)
    if anot:
        fig.suptitle(title)
    x = range(1, len(df_left) + 1)
    # plot the traces
    if left_sig:
        traces = [item for item in left.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edge_color = colors[i]
            color_dic = {0 : 'w', 1 : edge_color}
            sig_name = name + '_sig'
            select = df_left[[name, sig_name]].sort_values(by=[name, sig_name],
                                                           ascending=False)
            bar_colors = [color_dic[x] for x in select[sig_name]]
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            # without significance
            #select = df_left[name].sort_values(ascending=False)
            #ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
            #       alpha=0.8, width=0.8)
            # # with significance
            select = df_left[name].sort_values(ascending=False)

            ax.bar(x, select, color=bar_colors, edgecolor=edge_color,
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anot_left)
    else:
        for i, name in enumerate(df_left.columns):
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            # without significance
            select = df_left[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.5, width=0.8)
            if i == 0:
                ax.set_title(anot_left)
    #right
    if right_sig:
        traces = [item for item in right.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edge_color = colors[i]
            color_dic = {0 : 'w', 1 : edge_color}
            sig_name = name + '_sig'
            select = df_right[[name, sig_name]].sort_values(by=[name, sig_name],
                                                            ascending=False)
            bar_colors = [color_dic[x] for x in select[sig_name]]
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=bar_colors, edgecolor=edge_color,
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anot_right)
    else:
        for i, name in enumerate(df_right.columns):
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name , alpha=0.5)
            # without significance
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.5, width=0.8)
            if i == 0:
                ax.set_title(anot_right)

    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[:4]
    right_axes = axes[4:]
    for axe in [left_axes, right_axes]:
        for i, ax in enumerate(axe):
            ax.set_facecolor('None')
            # ax.set_title(i)
            for spine in ['top', 'bottom']:
                ax.spines[spine].set_visible(False)
           # ax.ticklabel_format(useOffset=True)
            # zero line
            ax.set_xlim(0, len(df_left)+1)
            lims = ax.get_xlim()
            ax.hlines(0, lims[0], lims[1], alpha=0.2)
            # ticks and ticks labels on both sides (call)
            # set_ticks_both(ax.yaxis)
            # alternate right and left
            # if overlap:
            #     #label left:
            #     if i % 2 == 0:
            #         ax.spines['right'].set_visible(False)
            #     #label right
            #     else:
            #         ax.spines['left'].set_visible(False)
            #         ax.yaxis.tick_right()
            # else:
            #     ax.spines['right'].set_visible(False)
            if i != 3:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel(anotx)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xticks([1, len(df_right)])
    # for ax in left_axes:
    #     custom_ticks = np.linspace(0, 0.5, 2)
    #     ax.set_yticks(custom_ticks)
    # for ax in right_axes:
    #     custom_ticks = np.linspace(0, 0.5, 2)
    #     ax.set_yticks(custom_ticks)
    no_spines = True
    if no_spines == True:
        for ax in left_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -10, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
        for ax in right_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -0.5, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
                ax.set_xlim(0, len(df_right)+1)

    # for ax in left_axes:
    #     custom_ticks = np.linspace(-10, 10, 3, dtype=int)
    #     ax.set_yticks(custom_ticks)
    # for ax in right_axes:
    #     ax.set_ylim(-0.5, 0.5)
    #     custom_ticks = np.linspace(-0.5, 0.5, 3)
    #     ax.set_yticks(custom_ticks)

    # align each row yaxis on zero between subplots
    # align_yaxis(left_axes[0], 0, right_axes[0], 0)
    # keep data range whithout distortion, preserve 0 alignment
    # change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots

    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.5, wspace=0.2)
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_sorted_responses',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def select_50(df, spread='sect', param='gain', noSig=True):
    """
    return the selected df parts for plotting
    spread in ['sect', 'full'],
    param in ['time', 'gain']
    """
    if spread not in ['sect', 'full']:
        print("'spread' should be in ['sect', 'full']")
        return
    if param not in ['time', 'gain', 'engy']:
        print("'param' should be in ['time', 'gain', 'engy']")
        return
    # select by param (first value = control)
    col_list = [item for item in df.columns if param in item]
    # select by spread
    col_list = [item for item in col_list if spread in item]
    if noSig:
        # remove sig
        col_list = [item for item in col_list if 'sig' not in item]
    return df[col_list]


def horizontal_dot_plot(df_left, df_right, mes=''):

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_responses' + ' (' + mes + ' ' + spread + ')'
    anoty = 'Cell rank'
    # anotx = [df_left.columns[0][5:], df_right.columns[0][5:]]
    anotx = [left.columns[0].split('_')[1], right.columns[0].split('_')[1]]
    fig = plt.figure()
    fig.suptitle(title)
    # left
    ax = fig.add_subplot(121)
    df = df_left.sort_values(by=df_left.columns[0], ascending=True)
    val_cols = [item for item in df.columns if '_sig' not in item]
    sig_cols = [item for item in df.columns if '_sig' in item]
#    df = df[cols]
    #sort
    sorted_cells = df.index.copy()
    for i, col in enumerate(val_cols):
        #define colors
        # marker_edgecolor = colors[i]
        # marker_facecolor =  if stat colors[i] else 'w'
        z1 = []
        for j in np.arange(len(df)):
            z1.append(colors[i])
        #stat
        z2 = []
        if col + '_sig' in df.columns:
            alpha = 0.8
            for a, b in zip(z1, df[col + '_sig']):
                if b == 1:
                    z2.append(a)
                else:
                    z2.append('w')
        else:
            alpha = 0.5
            z2 = z1
        #plot
        yl = df.index.to_list()
        xl = df[col].tolist()
        for x, y, e, f in zip(xl, yl, z1, z2):
            ax.plot(x, y, 'o', markeredgecolor=e, markerfacecolor=f, 
                    alpha=alpha, markeredgewidth=1.5, markersize=6)      
  # right
    ax = fig.add_subplot(122)
    df = df_right.reindex(sorted_cells)
    val_cols = [item for item in df.columns if '_sig' not in item]
    sig_cols = [item for item in df.columns if '_sig' in item]
    for i, col in enumerate(val_cols):
        #define colors
        # marker_edgecolor = colors[i]
        # marker_facecolor =  if stat colors[i] else 'w'
        z1 = []
        for j in np.arange(len(df)):
            z1.append(colors[i])
        #stat
        z2 = []
        if col + '_sig' in df.columns:
            alpha = 0.8
            for a, b in zip(z1, df[col + '_sig']):
                if b == 1:
                    z2.append(a)
                else:
                    z2.append('w')
        else:
            alpha = 0.5
            z2 = z1
        #plot
        yl = df.index.to_list()
        xl = df[col].tolist()
        for x, y, e, f in zip(xl, yl, z1, z2):
            ax.plot(x, y, 'o', markeredgecolor=e, markerfacecolor=f, 
                    alpha=alpha, markeredgewidth=1.5, markersize=6)      
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xlabel(anotx[i])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)
        ax.set_yticks([0, len(df) - 1])
        ax.set_yticklabels([1, len(df)])
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'toPlay.py:horizontal_dot_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def scatter_lat_gain(df_left, df_right, mes=''):
    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'responses' + ' (' + mes + ' ' + spread + ')'
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    # remove sig columns
    cols = [item for item in df_left.columns if '_sig' not in item]
    df_left = df_left[cols]
    cols = [item for item in df_right.columns if '_sig' not in item]
    df_right = df_right[cols]
    for i in range(len(df_left.columns)):
        color_list = []
        for j in range(len(df_left)):
            color_list.append(colors[i])
        x = df_left[df_left.columns[i]]
        y = df_right[df_right.columns[i]]
        ax.scatter(x, y,
                   c=color_list, s=100,
                   edgecolors=dark_colors[i], alpha=0.6)
    ax.set_xlabel('time')
    ax.set_ylabel('gain')
    ax.set_xlabel(df_left.columns[0].split('_')[1])
    ax.set_ylabel(df_right.columns[0].split('_')[1])
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], 'k', alpha=0.3)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'toPlay.py:horizontal_dot_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def histo_lat_gain(df_left, df_right, mes=''):
    """
    histogramme des données
    """
    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anoty = [df_left.columns[0].split('_')[1], 
             df_right.columns[0].split('_')[1]]
    # anoty = ['Relative peak advance(ms)', 'Relative peak amplitude']
    #          #(fraction of Center-only response)']
    # plot
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 2)
    # remove sig columns
    cols = [item for item in df_left.columns if '_sig' not in item]
    df_left = df_left[cols]
    cols = [item for item in df_right.columns if '_sig' not in item]
    df_right = df_right[cols]
    # left
    left_axes = []
    ax = fig.add_subplot(gs[0, 0])
    left_axes.append(ax)
    for i in range(1, 4):
        left_axes.append(fig.add_subplot(gs[i, 0], sharex=ax, sharey=ax))
    # right
    right_axes = []
    ax = fig.add_subplot(gs[0, 1], sharey=ax)
    right_axes.append(ax)
    for i in range(1, 4):
        right_axes.append(fig.add_subplot(gs[i, 1], sharex=ax, sharey=ax))
    # to identify the plots (uncomment to use)
    if anot:
        fig.suptitle(title)
    # plot the traces
    # left
    for i, name in enumerate(df_left.columns):
        ax = left_axes[i]
        # ax.set_title(str(i))
        # ax.set_title(name)
        ax.hist(df_left[name], bins=15, color=colors[i],
                edgecolor=dark_colors[i], alpha=0.8)
        # ax.hist(df_left[name], width=2, color=colors[i],
        #         edgecolor=dark_colors[i], alpha=0.8)
        if i == 0:
            ax.set_title(anoty[i])
    for i, name in enumerate(df_left.columns):
        ax = left_axes[i]
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=1)
        med = df_left[name].median()
        ax.vlines(med, lims[0], lims[1], color=dark_colors[i],
                  linewidth=2, linestyle=':')
        tx = 'med=' + str(round(med, 2))
        ax.text(0.65, 0.8, tx, horizontalalignment='left',
                transform=ax.transAxes, alpha=0.5)
        # right
    for i, name in enumerate(df_right.columns):
        ax = right_axes[i]
        # ax.set_title(str(i))
        # ax.set_title(name)
        ax.hist(df_right[name], bins=15, color=colors[i],
                edgecolor=dark_colors[i], alpha=0.8)
        # ax.hist(df_right[name], width = 0.05, color=colors[i],
        #         edgecolor=dark_colors[i], alpha=0.8)
        if i == 0:
            ax.set_title(anoty[1])
    for i, name in enumerate(df_right.columns):
        ax = right_axes[i]
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=1)
        med = df_right[name].median()
        ax.vlines(med, lims[0], lims[1], color=dark_colors[i],
                  linewidth=2, linestyle=':')
        tx = 'med=' + str(round(med, 2))
        ax.text(0.65, 0.8, tx, horizontalalignment='left',
                transform=ax.transAxes, alpha=0.5)

    for ax in fig.get_axes():
        ax.set_facecolor('None')
        ax.yaxis.set_visible(False)
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:histo_lat_gain',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def plot_glob_varia(left, right, mes=''):
    pass


def extract_stat(onlySig=False):
    # output
    desc_df = pd.DataFrame()
    # vm
    mes = 'vm'
    filename = 'data/cg_peakValueTime_vm.xlsx'
#    data50 = ldat.load_50vals(mes)
#    data50 = ldat.load_cell_contributions(mes)
    # amp in ['gain', 'engy']
    amp = 'gain'
    data50 = ldat.load_cell_contributions(kind=mes, amp=amp)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    print ('{} cells ({}, {})'.format(len(data50), mes, amp ))
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    sigCells = data50.index.to_list()
    # vm_time50
    times = [item for item in data50.columns if 'time' in item]
    times = [item for item in times if 'sect_' in item] \
            + [item for item in times if 'full_' in item]
    advance_df = data50[times].copy()
    advance_df.columns = [item.split('_')[0] for item in advance_df.columns]
    desc_df['time50_vm_mean'] = advance_df.mean()
    desc_df['time50_vm_std'] = advance_df.std()
    desc_df['time50_vm_med'] = advance_df.median()
    desc_df['time50_vm_mad'] = advance_df.mad()
    # vm_gain50
    gains = [item for item in data50.columns if 'gain' in item]
    gains = [item for item in gains if 'sect_' in item] \
            + [item for item in gains if 'full_' in item]
    gain_df = data50[gains].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['gain50_vm_mean'] = gain_df.mean()
    desc_df['gain50_vm_std'] = gain_df.std()
    desc_df['gain50_vm_med'] = gain_df.median()
    desc_df['gain50_vm_mad'] = gain_df.mad()
    # spike
    mes = 'spk'
    filename = 'data/cg_peakValueTime_spk.xlsx'
    data50 = ldat.load_cell_contributions(mes)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    # spk_time50
    times = [item for item in data50.columns if 'time' in item]
    times = [item for item in times if 'sect_' in item] \
            + [item for item in times if 'full_' in item]
    time_df = data50[times].copy()
    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    desc_df['time50_spk_mean'] = time_df.mean()
    desc_df['time50_spk_std'] = time_df.std()
    desc_df['time50_spk_med'] = time_df.median()
    desc_df['time50_spk_mad'] = time_df.mad()
    # spk_gain50
    gains = [item for item in data50.columns if 'gain' in item]
    gains = [item for item in gains if 'sect_' in item] \
            + [item for item in gains if 'full_' in item]
    gain_df = data50[gains].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['gain50_spk_mean'] = gain_df.mean()
    desc_df['gain50_spk_std'] = gain_df.std()
    desc_df['gain50_spk_med'] = gain_df.median()
    desc_df['gain50_spk_mad'] = gain_df.mad()
    # engy vm
    mes = 'vm'
    amp = 'engy'
    data50 = ldat.load_cell_contributions(kind=mes, amp=amp)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    print ('{} cells ({}, {})'.format(len(data50), mes, amp ))
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    sigCells = data50.index.to_list()
    # vm_gain50
    engy = [item for item in data50.columns if 'engy' in item]
    engy = [item for item in engy if 'sect_' in item] \
            + [item for item in engy if 'full_' in item]
    gain_df = data50[engy].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['energy_vm_mean'] = gain_df.mean()
    desc_df['energy_vm_std'] = gain_df.std()
    desc_df['energy_vm_med'] = gain_df.median()
    desc_df['energy_vm_mad'] = gain_df.mad()
    #engy spk
    mes = 'spk'
    amp = 'engy'
    data50 = ldat.load_cell_contributions(kind=mes, amp=amp)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    print ('{} cells ({}, {})'.format(len(data50), mes, amp ))
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    sigCells = data50.index.to_list()
    # vm_gain50
    engy = [item for item in data50.columns if 'engy' in item]
    engy = [item for item in engy if 'sect_' in item] \
            + [item for item in engy if 'full_' in item]
    gain_df = data50[engy].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['energy_spk_mean'] = gain_df.mean()
    desc_df['energy_spk_std'] = gain_df.std()
    desc_df['energy_spk_med'] = gain_df.median()
    desc_df['energy_spk_mad'] = gain_df.mad()

    # peak (non normalized data)
    # vm
    filename = 'data/cg_peakValueTime_vm.xlsx'
    data = load_peakdata(filename)
    gains = [item for item in data.columns if 'gain' in item]
    times = [item for item in data.columns if 'time' in item]
    # normalise
    for item in gains[1:]:
        data[item] = data[item] / data[gains[0]]
    # select
    gain_df = data[gains[1:]].copy()
    time_df = data[times[1:]].copy()
    #stat
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['gainP_vm_mean'] = gain_df.mean()
    desc_df['gainP_vm_std'] = gain_df.std()
    desc_df['gainP_vm_med'] = gain_df.median()
    desc_df['gainP_vm_mad'] = gain_df.mad()

    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    # time_df.rename(columns={'rndsect':'rndisosect', 'cpisosfull':'cpisofull',
    #                         'rndfull':'rndisofull'}, inplace=True)
    desc_df['timeP_vm_mean'] = time_df.mean()
    desc_df['timeP_vm_std'] = time_df.std()
    desc_df['timeP_vm_med'] = time_df.median()
    desc_df['timeP_vm_mad'] = time_df.mad()
    # spk_time
    filename = 'data/cg_peakValueTime_spk.xlsx'
    data = load_peakdata(filename)
    gains = [item for item in data.columns if 'gain' in item]
    times = [item for item in data.columns if 'time' in item]
    # normalise
    for item in gains[1:]:
        data[item] = data[item] / data[gains[0]]
    # select
    gain_df = data[gains[1:]].copy()
    time_df = data[times[1:]].copy()
    # stat
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['gainP_spk_mean'] = gain_df.mean()
    desc_df['gainP_spk_std'] = gain_df.std()
    desc_df['gainP_spk_med'] = gain_df.median()
    desc_df['gainP_spk_mad'] = gain_df.mad()

    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    desc_df['timeP_spk_mean'] = time_df.mean()
    desc_df['timeP_spk_std'] = time_df.std()
    desc_df['timeP_spk_med'] = time_df.median()
    desc_df['timeP_spk_mad'] = time_df.mad()

    return desc_df


plt.close('all')

def plot_stat(stat_df, kind='mean', loc='50'):
    """
    plot the stats
    input : stat_df, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    """
    if kind == 'mean':
        stat = ['_mean', '_std']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return
    if loc == '50':
        mes = ['time50', 'gain50']
    elif loc == 'peak':
        mes = ['timeP', 'gainP']
    elif loc == 'energy':
        mes = ['time50', 'energy']        
    else:
        print('non valid loc argument')
        return

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    fig = plt.figure(figsize=(8, 8))
    title = stat[0] + stat[1] + '\n' + mes[0] + mes[1]
    fig.suptitle(title)
    # sect vm
    axes = []
    ax = fig.add_subplot(221)
    axes.append(ax)
    ax1 = fig.add_subplot(2, 2, 2, sharex=ax, sharey=ax)
    axes.append(ax1)
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax)
    axes.append(ax2)
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax, sharey=ax2)
    axes.append(ax3)
    # vm
    ax.set_title('vm, sector')
    vals = [item for item in stat_df.columns if '_vm' in item]
    # sector
    spread = [item for item in stat_df.index if 'sect' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
         and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
         and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # full
    ax = axes[1]
    ax.set_title('vm, full')
    spread = [item for item in stat_df.index if 'full' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # spikes
    ax = axes[2]
    ax.set_title('spk, sector')
    vals = [item for item in stat_df.columns if '_spk' in item]
    # sector
    spread = [item for item in stat_df.index if 'sect' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # full
    ax = axes[3]
    ax.set_title('spk, full')
    spread = [item for item in stat_df.index if 'full' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]
    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)

    for i, ax in enumerate(axes):
        # lims = ax.get_ylim()
        # ax.vlines(0, lims[0], lims[1], linestyle=':', alpha=0.3)
        # lims = ax.get_xlim()
        # ax.hlines(0, lims[0], lims[1], linestyle=':', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel('gain')
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            if i > 1:
                ax.set_xlabel('time advance (ms)')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.xaxis.set_visible(False)
    # ax = axes[2]
    # custom_ticks = np.linspace(-2, 2, 3, dtype=int)/10
    # ax.set_yticks(custom_ticks)
    # ax.set_yticklabels(custom_ticks)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'toPlay.py:plot_stat',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

# to have sig and non sig:
def sigNonSig_stat_plot():
    """ stats for sig and non sig cells """
    stat_df = extract_stat(onlySig=False)
    fig0 = plot_stat(stat_df, 'med', '50')
    stat_df = extract_stat(onlySig=True)
    fig1 = plot_stat(stat_df, 'med', '50')
    axes0 = fig0.get_axes()
    axes1 = fig1.get_axes()
    ax0 = axes0[0]
    ax1 = axes1[0]
    leg = '37 cells'
    ax0.text(0.8, 0.8, leg, transform=ax0.transAxes)
    leg = '10 cells'
    ax1.text(0.8, 0.8, leg, transform=ax1.transAxes)
    for ax in [ax0, ax1]:
        ax.set_ylim(-0.15, 0.5)
    ax0 = axes0[2]
    ax1 = axes1[2]
    for ax in [ax0, ax1]:
        ax.set_ylim(-0.16, 0.75)
        ax.set_xlim(-12, 20)
    # zeroline
    alpha = 0.3
    for ax in fig0.get_axes():
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=alpha)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=alpha)
    for ax in fig1.get_axes():
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=alpha)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=alpha)


stat_df = extract_stat(onlySig=False)
# mind : significant cells are only base on cpisosect_time50 !
plot_stat(stat_df, 'med', 'peak')
plot_stat(stat_df, 'med', '50')
plot_stat(stat_df, 'med', 'energy')

# savePath = '/Users/cdesbois/ownCloud/cgFigures/pythonPreview/proposal/enerPeakOrGain'
# for measure in ['peak', '50', 'energy']:
#     fig = plot_stat(stat_df, 'med', measure)
#     fig.savefig(os.path.join(savePath, measure + '.png'))


#%% vm
plt.close('all')
for spread in ['sect', 'full']:
    mes = 'vm'
    data50 = ldat.load_cell_contributions(kind=mes, amp='gain', age='new')
    advance_df = select_50(data50, spread=spread, param='time', noSig=False)
    left = advance_df

    filename = 'data/cg_peakValueTime_vm.xlsx'
    data = load_peakdata(filename)
    right = normalize_peakdata_and_select(data.copy(), spread=spread, param='gain')

    # test for same cells
    diff = (left.index ^ right.index).tolist()
    if len(diff) > 0:
        print("left and right doesn't contain the same cells !")
        print(diff)
    for item in diff:
        if item in left.index:
            left = left.drop(item)
        elif item in right.index:
            right = right.drop(item)


    fig = plot_sorted_responses(left, right, mes=mes, overlap=True,
                                left_sig=True, right_sig=False)
    fig2 = horizontal_dot_plot(left, right, mes=mes)
    fig3 = scatter_lat_gain(left, right, mes=mes)
    fig4 = histo_lat_gain(left, right, mes=mes)

#%% spk
for spread in ['sect', 'full']:
    mes = 'spk'
    data50 = ldat.load_cell_contributions(mes)
    advance_df = select_50(data50, spread=spread, param='time', noSig=False)
    left = advance_df

    filename = 'data/cg_peakValueTime_spk.xlsx'
    data = load_peakdata(filename)
    right = normalize_peakdata_and_select(data.copy(), spread=spread, param='gain')

    # test for same cells
    diff = (left.index ^ right.index).tolist()
    if len(diff) > 0:
        print("left and right doesn't contain the same cells !")
        print(diff)
    for item in diff:
        if item in left.index:
            left = left.drop(item)
        elif item in right.index:
            right = right.drop(item)

    fig = plot_sorted_responses(left, right, mes=mes, overlap=True,
                                left_sig=True, right_sig=False)
    fig2 = horizontal_dot_plot(left, right, mes=mes)
    fig3 = scatter_lat_gain(left, right, mes=mes)
    fig4 = histo_lat_gain(left, right, mes=mes)

#%% energy
plt.close('all')
for spread in ['sect', 'full']:
    mes = 'vm'
    data50 = ldat.load_cell_contributions(kind=mes, amp='gain', age='new')
    left = select_50(data50, spread=spread, param='gain', noSig=False)

    data50 = ldat.load_cell_contributions(kind=mes, amp='engy', age='new')
    right = select_50(data50, spread=spread, param='engy', noSig=False)

    fig = plot_sorted_responses(left, right, mes=mes, overlap=True,
                                left_sig=True, right_sig=True)
    fig2 = horizontal_dot_plot(right, left, mes=mes)
    fig3 = scatter_lat_gain(left, right, mes=mes)
    fig4 = histo_lat_gain(left, right, mes=mes)
#%% plot energy vm
def adapt_energy_to_plot(energy_df, spread='sect'):
    df = energy_df.copy()
    #remove stats
#    cols = [col for col in df.columns if '_p' not in col]
    #select sector
    ctr = df['centeronly_energy'].copy()
    cols = [col for col in df.columns if spread[:3] in col]
    df = df[cols].copy()
    #normalize
    traces = [col for col in cols if '_sig' not in col]
    for col in traces:
        df[col] = (df[col] - ctr) / ctr
    return df

spread = 'sect'
mes = 'vm'
data50 = ldat.load_cell_contributions(mes)
gain_df = select_50(data50, spread=spread, param='gain', noSig=False)
left = gain_df

paths = config.build_paths()
energy_df = ldat.load_energy_gain_index(paths)
right = adapt_energy_to_plot(energy_df)
fig = plot_sorted_responses(left, right, mes=mes, overlap=True)


#%% test to rebuild sup1 using horizontal dot  plot
plt.close('all')

for mes in ['vm', 'spk']:
    for spread in ['sect', 'full']:
        # nb builded on data/figSup34Vm.xlsx
        data_df = ldat.load_cell_contributions(mes)
        traces = [item for item in data_df.columns if spread in item]
        # remove random sector
        traces = [item for item in traces if 'rdisosect' not in item]
        # append rdiso
        traces += [item for item in data_df.columns if 'rdisofull' in item]
        #remove duplicate 
        traces = list(dict.fromkeys(traces))
        # filter -> only significative cells
        #traces = [item for item in traces if 'sig' not in item]
        left = data_df[[item for item in traces if 'time50' in item]]
        right = data_df[[item for item in traces if 'gain50' in item]]

        fig = horizontal_dot_plot(left, right, mes=mes)
        fig.suptitle(mes + '  ' +  spread)
        axes = fig.get_axes()
        ax = axes[0]
        ax.set_ylabel('', rotation=0)

        #to label cells
        # cells = left.sort_values(left.columns[0], ascending=True).index.tolist()
        # names = [item.split('_')[0] for item in cells]
        # ax.set_yticks(np.arange(len(cells)))
        # ax.set_yticklabels(names, fontsize=8)
        # lims = ax.get_xlim()
        # from math import ceil
        # ax.set_xlim(ceil(lims[1]), ceil(lims[0]))
        # ax.set_xlim(15, -30)
    
        # ax = axes[1]
        # ax.set_yticks(np.arange(len(cells)) + 1)
        # ax.set_yticklabels('', fontsize=8)
        
        fig.tight_layout()

#TODO : implement the siginicativity & add the spikes
